import argparse

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers import logging as transformers_logging
from datasets import load_dataset
import torch
import os
import logging
import multiprocess


from functools import partial

# Configure logging
logger = logging.getLogger(__name__)

# Only show errors, hide warnings
transformers_logging.set_verbosity_error()

ENTITY_TYPES = ['Miscellaneous', 'Blockchain_Name', 'Codebase',
                'ChargingAndRewardingSystem', 'ESG', 'Extensibility',
                'Identifiers', 'Identity_Management', 'Native_Currency_Tokenisation',
                'Security_Privacy', 'Transaction_Capabilities', 'Consensus']


def init_model(model_dir, device=None):

    if device is not None:
        model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)

    else:
        # Load the model using device_map='auto' to automatically distribute the model across available devices
        model = AutoModelForTokenClassification.from_pretrained(model_dir, device_map = 'auto')

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    return model, tokenizer


def inference(model, tokenizer, text, chunking_enabled=True, aggregation_strategy="first", window_size=512,
              device=None):
    # Create the NER pipeline
    if device:
        ner = pipeline("ner", model=model, tokenizer=tokenizer,
                       aggregation_strategy=aggregation_strategy,
                       device=device)
    else:
        ner = pipeline("ner", model=model, tokenizer=tokenizer,
                       aggregation_strategy=aggregation_strategy, device_map="auto")

    # Encode the full text to get token IDs for chunking into windows
    # Note: We use truncation=False to avoid truncating the input text
    # This is important for chunking logic
    # The way it is would give the warning
    # "Token indices sequence length is longer than the specified maximum sequence length"
    # However, the warning is not an error and the model will still work
    # The warning is just to inform that the input text is longer than the model's max length
    # We can ignore it and the warnings has been disabled for this script
    input_ids = tokenizer.encode(text, truncation=False)

    # Check if we need chunking
    if chunking_enabled and len(input_ids) > window_size:
        # Split the sequences into chunks of window_size tokens
        chunks = [input_ids[i:i + window_size] for i in range(0, len(input_ids), window_size)]

        # Perform inference on each chunk
        predictions = []
        for chunk in chunks:
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if chunk_text.strip():  # Ensure chunk is not just whitespace
                chunk_preds = ner(chunk_text)
                predictions.extend(chunk_preds)
    else:
        # For short texts, perform inference directly
        predictions = ner(text)

    return predictions


def process_text(example, model, tokenizer, window_size=512, device=None, aggregation_strategy="first"):
    text = example['text']
    predictions = inference(model, tokenizer, text, chunking_enabled=True,
                            aggregation_strategy=aggregation_strategy, window_size=window_size, device=device)

    # Add entity groups to example
    result = {**example}

    result['predictions'] = predictions

    return result


def process_example_on_gpu(example, rank, model_dir_path, total_gpus,
                           aggregation_strategy: str ="first", window_size: int =512):
    """
    Worker function to process a single example on an assigned GPU.
    This function is called by datasets.map() for each example.
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

    processed_result = process_text(example, model, tokenizer, window_size=window_size,
                                    device=model_target_device, aggregation_strategy=aggregation_strategy)

    return processed_result


if __name__ == "__main__":

    # IMPORTANT: Set the start method to 'spawn' for CUDA multiprocessing compatibility
    # This should be done at the very beginning of the if __name__ == "__main__": block.
    # And before any CUDA or multiprocessing related initializations.
    # `force=True` is used to ensure it's set even if it was implicitly set before.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        multiprocess.set_start_method("spawn", force=True)
        logger.info("Set multiprocessing start method to 'spawn'.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default=None,
                        help="Directory of the fine-tuned model and tokenizer.")
                        
    parser.add_argument("--hf_dataset", type=str,
                        default="ExponentialScience/DLT-Open-Access-Literature-annotated-filtered",
                        help="Hugging Face dataset name to process.")

    parser.add_argument("--output_hf_dataset", type=str,
                        default="ExponentialScience/DLT-Open-Access-Literature-annotated-entities",
                        help="Name for the output dataset on Hugging Face Hub.")
    
    parser.add_argument('--hf_datasets', type=str,
                        default="./../../../data/hf_datasets",
                        help='Path to save the Hugging Face dataset in disk.')

    parser.add_argument("--add_special_tokens", action='store_true', default=False,
                        help="Whether to add special tokens during tokenization (passed to AutoTokenizer).")

    parser.add_argument("--aggregation_strategy", type=str, default="first",
                        help="Aggregation strategy for the NER pipeline (e.g., 'simple', 'first', 'average', 'max').")

    parser.add_argument("--window_size", type=int, default=512,
                        help="Window size for text chunking. Set to 0 or negative to disable chunking.")

    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Maximum sequence length for tokenizer")

    parser.add_argument("--numb_gpus_use", type=int, default=1,
                        help="Set the number of GPUs required to use.")

    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # 
    # Load the dataset
    hf_dataset = load_dataset(args.hf_dataset, split="train")
    # To test with a smaller subset:
    # hf_dataset = hf_dataset.shard(num_shards=1000, index=0)

    # df = pd.read_csv("./../../../models/dlt_annotated.csv")
    #
    # ## Convert pandas dataframe to Hugging Face dataset
    # hf_dataset = Dataset.from_pandas(df)

    columns = hf_dataset.column_names
    original_text_col_name = None
    if "tweet" in columns:
        original_text_col_name = "tweet"
        logger.info(f"Renaming 'tweet' column to 'text'")
        hf_dataset = hf_dataset.rename_column("tweet", "text")
    
    num_cuda_gpus = torch.cuda.device_count()

    # Determine the number of processes for map
    # Use all available CUDA GPUs. If none, use 1 process (for CPU or MPS).
    num_processes_for_map = num_cuda_gpus if num_cuda_gpus > 0 else 1
    
    logger.info(f"Available CUDA GPUs: {num_cuda_gpus}")
    if num_cuda_gpus == 0:
        if torch.backends.mps.is_available():
            logger.info("No CUDA GPUs found. MPS is available. Will attempt to use MPS if applicable, or CPU.")
        else:
            logger.info("No CUDA GPUs found. Processing will use CPU.")

    logger.info(f"Using {num_processes_for_map} worker process(es) for dataset mapping.")


    if num_cuda_gpus > 1 and args.numb_gpus_use > 1:
        # Create a partial function with fixed arguments for the worker
        # The 'rank' argument will be supplied by `map` when `with_rank=True`
        map_worker_fn = partial(
            process_example_on_gpu,
            model_dir_path=args.model_dir,
            total_gpus=num_cuda_gpus,  # Pass the count of CUDA GPUs
            aggregation_strategy=args.aggregation_strategy,
            window_size=args.window_size
        )

        # Apply the processing function using map
        # `with_rank=True` provides the 'rank' argument to `map_worker_fn`
        logger.info("Starting dataset processing...")
        processed_dataset = hf_dataset.map(
            map_worker_fn,
            with_rank=True,
            num_proc=num_processes_for_map)

    else:
        model, tokenizer = init_model(args.model_dir)

        # set model_max_length to 512 as label texts are no longer than 512 tokens
        if args.max_seq_length is not None:
            tokenizer.model_max_length = args.max_seq_length
        else:
            args.max_seq_length = tokenizer.model_max_length

        # Create a partial function with model and tokenizer
        process_func = partial(process_text, model=model, tokenizer=tokenizer)

        # ## Convert pandas dataframe to Hugging Face dataset
        # hf_dataset = Dataset.from_pandas(df)

        # Apply processing in parallel with multiple processors
        # set_start_method("spawn")
        processed_dataset = hf_dataset.map(
            process_func,
            # with_rank=True,
            # num_proc=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        )
    
    # Save the Hugging Face dataset
    hf_dir = os.path.join(args.hf_datasets)
    os.makedirs(hf_dir, exist_ok=True)
    hf_dataset_path = os.path.join(hf_dir, "ner_processed")

    # rename the text column back to original if needed
    if original_text_col_name is not None:
        logger.info(f"Renaming 'text' column back to '{original_text_col_name}'")
        processed_dataset = processed_dataset.rename_column("text", original_text_col_name)

    logger.info(f"Writing HF dataset locally to {hf_dataset_path}")
    processed_dataset.save_to_disk(hf_dataset_path)
    
    # Push the processed dataset to Hugging Face Hub
    logger.info(f"Pushing processed dataset to Hugging Face Hub: {args.output_hf_dataset}...")
    processed_dataset.push_to_hub(args.output_hf_dataset, private=True)

    logger.info("Processing complete and dataset pushed to Hub.")



