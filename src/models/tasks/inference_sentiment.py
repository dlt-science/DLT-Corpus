import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import logging as transformers_logging
from datasets import load_dataset
import torch
import os
import logging
import multiprocess

from functools import partial

# Adding the path to be able to import the analytics module
import sys

sys.path.append('./../../')
from models.continual_training import load_dataset_safe

# Configure logging
logger = logging.getLogger(__name__)

# Only show errors, hide warnings
transformers_logging.set_verbosity_error()

# Define sentiment labels - can be customized based on model
SENTIMENT_LABELS = ['bearish', 'neutral', 'bullish']  # Common 3-class sentiment


def init_model(model_dir, device=None):
    """
    Initialize the sentiment analysis model and tokenizer.

    Args:
        model_dir (str): Path to the model directory
        device (str, optional): Device to load the model on ('cuda:0', 'cpu', 'mps', etc.)

    Returns:
        tuple: (model, tokenizer) loaded and ready for inference
    """
    if device is not None:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    else:
        # Load the model using device_map='auto' to automatically distribute across available devices
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, device_map='auto')

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    return model, tokenizer


def inference(model, tokenizer, text, top_k=None, window_size=512, device=None):
    """
    Perform sentiment analysis inference on input text.

    Args:
        model: The sentiment analysis model
        tokenizer: The tokenizer for the model
        text (str): Input text to analyze
        top_k (int): Number of top predictions to return (None for all, 1 for top prediction)
        window_size (int): Maximum window size for chunking long texts
        device (str, optional): Device to run inference on
        aggregation_method (str): Method to aggregate scores from chunks ('mean', 'max', 'weighted_mean')

    Returns:
        dict: Sentiment prediction with label and score, optionally all scores
    """
    # Create the sentiment analysis pipeline
    if device:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=top_k
        )
    else:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            top_k=top_k
        )

    # For short texts, perform inference directly
    predictions = sentiment_analyzer(text)

    if top_k is None:
        return predictions[0]

    return predictions


def process_text(example, model, tokenizer, window_size=512, device=None,
                 top_k=None, text_column="text"):
    """
    Process a single text example for sentiment analysis.

    Args:
        example (dict): Dataset example containing text
        model: The sentiment analysis model
        tokenizer: The tokenizer
        window_size (int): Maximum window size for chunking
        device (str): Device to run inference on
        top_k (int): Number of top predictions to return (None for all, 1 for top prediction)
        aggregation_method (str): Method to aggregate chunk scores
        text_column (str): Name of the column containing text

    Returns:
        dict: Original example with added sentiment predictions
    """
    text = example.get(text_column, "")

    if not text or not text.strip():
        # Handle empty text
        logger.warning(f"Empty text found in example")
        result = {**example}
        result['sentiment_label'] = None
        result['sentiment_score'] = None
        if top_k is None or top_k > 1:
            result['all_scores'] = []
        return result

    predictions = inference(
        model, tokenizer, text,
        top_k=top_k,
        window_size=window_size,
        device=device,
    )

    # Add predictions to the example
    result = {**example}

    result['sentiment_class'] = predictions

    # Also store the top prediction
    if predictions and len(predictions) > 0:
        result['sentiment_label'] = predictions[0]['label']
        result['sentiment_score'] = predictions[0]['score']

        # # Show real-time prediction example
        # logger.info(
        #     f"PREDICTION: '{predictions[0]['label']}' (score: {predictions[0]['score']:.4f}) | Text: {text[:100]}...")
    else:
        result['sentiment_label'] = None
        result['sentiment_score'] = None

    # Add confidence level based on score
    if result['sentiment_score'] is not None:
        if result['sentiment_score'] > 0.9:
            result['confidence_level'] = 'high'
        elif result['sentiment_score'] > 0.7:
            result['confidence_level'] = 'medium'
        else:
            result['confidence_level'] = 'low'
    else:
        result['confidence_level'] = None

    # # Show example of processed result
    # logger.debug(f"Processed example sentiment: {result['sentiment_label']} "
    #              f"({result['sentiment_score']:.4f}, {result['confidence_level']}) for text: {text[:50]}...")

    return result


def process_example_on_gpu(example, rank, model_dir_path, total_gpus,
                           window_size=512, text_column="text"):
    """
    Worker function to process a single example on an assigned GPU.
    This function is called by datasets.map() for each example when using multiple GPUs.

    Args:
        example (dict): Dataset example
        rank (int): Worker rank for GPU assignment
        model_dir_path (str): Path to the model
        total_gpus (int): Total number of available GPUs
        window_size (int): Maximum window size for chunking
        text_column (str): Name of the text column

    Returns:
        dict: Processed example with sentiment predictions
    """
    actual_device_id = 0
    model_target_device = 'cpu'  # Default to CPU if no GPUs

    if total_gpus > 0:
        actual_device_id = rank % total_gpus
        model_target_device = f'cuda:{actual_device_id}'
    elif torch.backends.mps.is_available():
        model_target_device = 'mps'
        actual_device_id = 'mps'  # For pipeline device argument
    else:
        actual_device_id = -1  # For pipeline device argument (CPU)

    # Load model and tokenizer inside the worker process for the assigned device
    model, tokenizer = init_model(model_dir_path, device=model_target_device)

    processed_result = process_text(
        example, model, tokenizer,
        window_size=window_size,
        device=model_target_device,
        text_column=text_column,
        top_k=None
    )

    return processed_result



if __name__ == "__main__":

    # IMPORTANT: Set the start method to 'spawn' for CUDA multiprocessing compatibility
    # This should be done at the very beginning of the if __name__ == "__main__": block.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        multiprocess.set_start_method("spawn", force=True)
        logger.info("Set multiprocessing start method to 'spawn'.")

    parser = argparse.ArgumentParser(description="Sentiment Analysis Inference on HuggingFace Datasets")

    parser.add_argument("--model_dir", type=str,
                        default="./../../../models/LedgerBERT-SciBERT-base-v1--News-Class/2025-09-10_15-44-05/market_direction",
                        help="Directory or HuggingFace ID of the sentiment analysis model.")

    parser.add_argument("--dataset_name", type=str,
                        default="ExponentialScience/Crypto-Tweets-NER",
                        help="Hugging Face dataset name to process.")

    parser.add_argument("--output_hf_dataset", type=str,
                        default="ExponentialScience/Crypto-Tweets-NER-Sentiment",
                        help="Name for the output dataset on Hugging Face Hub.")

    parser.add_argument('--hf_datasets', type=str,
                        default="./../../../data/hf_datasets",
                        help='Path to save the Hugging Face dataset on disk.')

    parser.add_argument("--text_column", type=str, default="text",
                        help="Name of the column containing text to analyze.")

    parser.add_argument("--window_size", type=int, default=512,
                        help="Window size for text chunking. Set to 0 to disable chunking.")

    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Maximum sequence length for tokenizer.")

    parser.add_argument("--numb_gpus_use", type=int, default=1,
                        help="Set the number of GPUs to use.")

    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for dataset mapping (number of examples per batch).")

    parser.add_argument("--push_to_hub", action='store_true', default=True,
                        help="Whether to push the processed dataset to Hugging Face Hub.")

    parser.add_argument("--private", action='store_true', default=True,
                        help="Whether to make the Hub dataset private.")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load the dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    hf_dataset = load_dataset_safe(args.hf_datasets, args.dataset_name)

    # # To test with a smaller subset (uncomment if needed):
    # hf_dataset = hf_dataset.select(range(100))  # Process only first 100 examples

    # Handle different column names for text
    columns = hf_dataset.column_names
    original_text_col_name = None

    # Common column name mappings
    text_column_mappings = {
        "tweet": "text",
        "review": "text",
        "comment": "text",
        "sentence": "text",
        "document": "text"
    }

    # Check if we need to rename any columns
    if args.text_column not in hf_dataset.column_names:
        for old_name, new_name in text_column_mappings.items():
            if old_name in columns and args.text_column == "text":
                original_text_col_name = old_name
                logger.info(f"Renaming '{old_name}' column to '{new_name}'")
                hf_dataset = hf_dataset.rename_column(old_name, new_name)
                break

    # Verify the text column exists
    if args.text_column not in hf_dataset.column_names:
        logger.error(f"Text column '{args.text_column}' not found in dataset columns: {hf_dataset.column_names}")
        raise ValueError(f"Text column '{args.text_column}' not found in dataset")

    # Determine available compute resources
    num_cuda_gpus = torch.cuda.device_count()

    # Determine the number of processes for map
    # Use requested GPUs if available, otherwise use 1 process (for CPU or MPS)
    num_processes_for_map = min(args.numb_gpus_use, num_cuda_gpus) if num_cuda_gpus > 0 else 1

    logger.info(f"Available CUDA GPUs: {num_cuda_gpus}")
    if num_cuda_gpus == 0:
        if torch.backends.mps.is_available():
            logger.info("No CUDA GPUs found. MPS is available. Will use MPS for processing.")
        else:
            logger.info("No CUDA GPUs found. Processing will use CPU.")

    logger.info(f"Using {num_processes_for_map} worker process(es) for dataset mapping.")

    # Process the dataset
    if num_cuda_gpus > 1 and args.numb_gpus_use > 1:
        # Multi-GPU processing
        logger.info("Using multi-GPU processing...")

        # Create a partial function with fixed arguments for the worker
        map_worker_fn = partial(
            process_example_on_gpu,
            model_dir_path=args.model_dir,
            total_gpus=num_cuda_gpus,
            window_size=args.window_size,
            text_column=args.text_column
        )

        # Apply the processing function using map with multiple processes
        logger.info("Starting dataset processing...")
        processed_dataset = hf_dataset.map(
            map_worker_fn,
            with_rank=True,
            num_proc=num_processes_for_map,
            batch_size=args.batch_size,
            desc="Processing sentiment analysis",
            load_from_cache_file=True  # Use cache to save processing time
        )
    else:
        # Single GPU/CPU/MPS processing
        logger.info("Using single device processing...")

        # Initialize model and tokenizer once
        model, tokenizer = init_model(args.model_dir)

        # Set model_max_length if specified
        if args.max_seq_length is not None:
            tokenizer.model_max_length = args.max_seq_length
        else:
            args.max_seq_length = tokenizer.model_max_length

        logger.info(f"Max sequence length: {args.max_seq_length}")

        # Create a partial function with model and tokenizer
        process_func = partial(
            process_text,
            model=model,
            tokenizer=tokenizer,
            window_size=args.window_size,
            text_column=args.text_column
        )

        # Apply processing (single process)
        logger.info("Starting dataset processing...")
        processed_dataset = hf_dataset.map(
            process_func,
            batch_size=args.batch_size,
            desc="Processing sentiment analysis",
            load_from_cache_file=True  # Use cache to save processing time
        )

    # Show 4 rows of the processed dataset
    logger.info("Sample processed examples:")
    df_example = processed_dataset.select(range(5)).to_pandas()

    for i in range(3):

        logger.info(
            f"PREDICTION: '{df_example.iloc[i]['sentiment_label']}' "
            f"(score: {df_example.iloc[i]['sentiment_score']:.4f}) "
            f"| Text: {df_example.iloc[i][args.text_column][:100]}...")

    # Save the processed dataset locally
    hf_dir = os.path.join(args.hf_datasets)
    os.makedirs(hf_dir, exist_ok=True)
    hf_dataset_path = os.path.join(hf_dir, "sentiment_processed")

    # Rename the text column back to original if needed
    if original_text_col_name is not None:
        logger.info(f"Renaming 'text' column back to '{original_text_col_name}'")
        processed_dataset = processed_dataset.rename_column("text", original_text_col_name)

    logger.info(f"Writing HF dataset locally to {hf_dataset_path}")
    processed_dataset.save_to_disk(hf_dataset_path)

    # Optionally push to Hugging Face Hub
    if args.push_to_hub:
        logger.info(f"Pushing processed dataset to Hugging Face Hub: {args.output_hf_dataset}...")
        processed_dataset.push_to_hub(args.output_hf_dataset, private=args.private)
        logger.info("Dataset successfully pushed to Hub.")

    # Print summary statistics
    logger.info("\n" + "=" * 50)
    logger.info("Processing Complete - Summary Statistics")
    logger.info("=" * 50)
    logger.info(f"Total examples processed: {len(processed_dataset)}")

    # Calculate sentiment distribution if available
    if 'sentiment_label' in processed_dataset.column_names:
        sentiment_counts = {}
        for example in processed_dataset:
            label = example.get('sentiment_label')
            if label:
                sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

        logger.info("\nSentiment Distribution:")
        for label, count in sorted(sentiment_counts.items()):
            percentage = (count / len(processed_dataset)) * 100
            logger.info(f"  {label}: {count} ({percentage:.2f}%)")

    logger.info("\nProcessing complete!")