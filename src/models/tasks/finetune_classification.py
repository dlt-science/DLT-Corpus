import os
import numpy as np
import pandas as pd
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
    EarlyStoppingCallback
)
import multiprocessing
import logging
import wandb

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
from functools import partial
from multiprocessing import cpu_count
import json

# Adding the path to be able to import the analytics module
import sys

sys.path.append('./../../')
from models.continual_training import find_latest_checkpoint, load_dataset_safe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Optimize for multi-core CPU usage
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)

# Load environment variables (for W&B API key, etc.)
from dotenv import load_dotenv
load_dotenv()

# Set W&B project name from environment variable
os.environ["WANDB_PROJECT"] = os.environ["WANDB_PROJECT_CLASSIFICATION"]


def compute_metrics(eval_pred):
    """
    Compute metrics for sentiment analysis evaluation.
    Returns accuracy, precision, recall, and F1 score.
    """
    predictions, labels = eval_pred

    # Keep raw probabilities for AUC calculation
    predictions = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Calculate precision, recall, F1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )

    # Calculate macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )

    # Build metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
    }

    return metrics

# def format_dataset(examples, dataset_name):
#     """Format data for engagement quality classification (neutral, liked, disliked)"""
#     return {
#         'text': [f"{title}\n\n{desc}".strip() for title, desc in zip(examples['title'], examples['description'])],
#         'labels': examples[dataset_name]
#     }


def preprocess_function(examples, tokenizer, max_length=512):

    # Truncates long texts to fit within the model's max sequence length
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)


def chunk_texts_with_labels(examples, tokenizer, max_length=512, overlap=50, label_column=None):
    """
    Chunk long texts into smaller segments while preserving labels.

    Args:
        examples: Dictionary with 'text' and label columns
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length for each chunk
        overlap: Number of tokens to overlap between chunks

    Returns:
        Dictionary with chunked texts and repeated labels
    """
    chunked_texts = []
    chunked_labels = []

    if label_column is None:
        # Get the label column name (could be 'labels', 'label', or custom column)
        label_column = [col for col in examples.keys() if col.lower() in ['label', 'labels']]
        if not label_column:
            raise ValueError("No label column found in examples")

        label_column = label_column[0]  # Use the first found label column

        if label_column is None:
            raise ValueError("No label column found in examples")

    for text, label in zip(examples['text'], examples[label_column]):
        # Tokenize the full text first
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # If text is shorter than max_length, keep it as is
        if len(tokens) <= max_length - 2:  # -2 for [CLS] and [SEP] tokens
            chunked_texts.append(text)
            chunked_labels.append(label)
        else:
            # Create overlapping chunks
            chunk_size = max_length - 2  # Account for special tokens
            start = 0

            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]

                # Convert back to text
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunked_texts.append(chunk_text)
                chunked_labels.append(label)

                # Move start position with overlap
                if end == len(tokens):
                    break
                start = end - overlap

    # Return the chunked data
    result = {'text': chunked_texts, label_column: chunked_labels}
    return result


def get_label_vals(formatted_ds):
    # Get the label2id from the dataset
    if 'labels' in formatted_ds.column_names:
        label_names = formatted_ds.features['labels'].names
        id2label = {i: name for i, name in enumerate(formatted_ds.features['labels'].names)}

    elif 'label' in formatted_ds.column_names:
        label_names = formatted_ds.features['label'].names
        id2label = {i: name for i, name in enumerate(formatted_ds.features['label'].names)}
    else:
        raise ValueError("Dataset does not contain 'labels' or 'label' column")

    label2id = {name: i for i, name in id2label.items()}

    logger.info(f"label2id: {label2id}")
    logger.info(f"id2label: {id2label}")
    logger.info(f"label_names: {label_names}")

    return label_names, label2id, id2label

def train_model(dataset_name, args, tokenizer, task_output_dir, col_labels = None, run_name=None):
    """
    Train a sentiment analysis model for a specific task.

    Args:
        dataset_name: Name of the sentiment task (e.g., 'engagement_quality')
        format_function: Function to format the dataset
        label_names: List of label names for this task
        args: Command line arguments
        tokenizer: Pre-loaded tokenizer
        task_output_dir: Base directory for outputs

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training model for: {dataset_name}")
    logger.info(f"{'=' * 60}")

    # Load and format dataset
    logger.info(f"Loading dataset from {args.dataset_name}")
    ds = load_dataset_safe(
        args.hf_datasets, args.dataset_name
    )

    # # Apply formatting function
    # format_function = partial(format_dataset, dataset_name=dataset_name)
    #
    # if args.dataset_name == "ExponentialScience/DLT-Sentiment-News":
    #     logger.info(f"Formatting dataset of {dataset_name}")
    #     ds = ds.map(format_function, batched=True)

    if col_labels:
        logger.info(f"Using custom label column: {col_labels} and renaming it to 'label'")
        ds = ds.rename_column(col_labels, 'label')
        col_labels = None

    # Remove unnecessary columns
    cols_to_keep = ['text', 'code', 'labels', 'label']

    cols_to_remove = [col for col in ds.column_names if col not in cols_to_keep]
    formatted_ds = ds.remove_columns(cols_to_remove)

    # Rename code column to text if it exists
    if 'code' in formatted_ds.column_names:
        formatted_ds = formatted_ds.rename_column('code', 'text')

    logger.info(f"Columns after formatting: {formatted_ds.column_names}")

    label_names, label2id, id2label = get_label_vals(formatted_ds)

    logger.info(f"Labels: {label_names}")

    # Calculate optimal process count (if not specified)
    if args.num_processes is None:
        num_processes = max(1, int(multiprocessing.cpu_count() * 2 / 3))
        logger.info(f"Using {num_processes} processes (2/3 of available {multiprocessing.cpu_count()} CPU cores)")
    else:
        num_processes = args.num_processes
        logger.info(f"Using {num_processes} processes as specified")


    #### Chunking disabled for now becase it is not needed for sentiment tasks with short texts ####
    # logger.info(f"Chunking texts to fit within model's max sequence length of {args.max_seq_length}")
    # chunk_function = partial(
    #     chunk_texts_with_labels,
    #     tokenizer=tokenizer,
    #     max_length=args.max_seq_length - 2,  # Account for special tokens
    #     overlap=100, # Overlap of 100 tokens between chunks for context
    #     label_column=col_labels  # Use specified label column if provided
    # )
    #
    # # Apply chunking
    # chunked_ds = formatted_ds.map(
    #     chunk_function,
    #     num_proc=num_processes,
    #     batched=True,
    #     load_from_cache_file=True,  # Use cache to save processing time
    # )

    # Tokenize the dataset
    # Pass multiple arguments to the preprocess function
    preprocesses_fn = partial(preprocess_function,
                              tokenizer=tokenizer,
                              max_length=args.max_seq_length
                              )

    logger.info(f"Tokenizing the dataset with {num_processes} processes")
    tok_ds = formatted_ds.map(preprocesses_fn,
                    num_proc=num_processes,
                    batched=True,
                              remove_columns=[col for col in formatted_ds.column_names if
                                              col not in ['labels', 'label', col_labels]],  # Keep labels column
                              load_from_cache_file=True,  # Use cache to save processing time
                    )

    logger.info(f"Train-test splitting the dataset with 10% for testing")

    stratify_col = 'labels' if 'labels' in tok_ds.column_names else 'label'
    if col_labels:
        stratify_col = col_labels

    train_test_dataset = tok_ds.train_test_split(test_size=args.test_size, seed=args.seed,
                                                            stratify_by_column=stratify_col)

    # Load model
    model_path = args.checkpoint_path if args.checkpoint_path else args.model_name

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logs_dir = os.path.join(task_output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    if not run_name:
        run_name = f"{args.model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=task_output_dir,
        num_train_epochs=args.num_epochs,

        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,

        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,

        do_train=True,
        do_eval=True,
        warmup_ratio = 0.1,
        weight_decay=0.1,
        label_smoothing_factor=0.1,  # ← Prevents overconfident predictions

        # Evaluation settings
        # eval_strategy="steps",

        # Need to match the save_strategy for correct functioning of load_best_model_at_end
        # when using "loss" for metric_for_best_model
        eval_strategy="epoch",
        # eval_strategy="steps",

        # eval_accumulation_steps=1,
        # eval_steps=10000,
        eval_on_start=True,

        # TODO: Enable loading of best model at the end of training
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        # metric_for_best_model="loss",
        # greater_is_better=True,

        # metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_dir=logs_dir,
        logging_strategy="steps",
        logging_first_step=True,  # Add this to log the first step
        logging_steps=10,
        report_to="wandb",  # enable logging to W&B

        run_name=run_name,

        # Memory optimization
        dataloader_num_workers=min(8, cpu_count() // 2),

        # allows to effectively have a larger batch size while using less memory per step
        # # Use bfloat16 for better memory efficiency
        # fp16=False if torch.backends.mps.is_available() else True,  # Enable mixed precision training
        # Enable bf16 if supported
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        bf16_full_eval=True,

        # Checkpoint management
        save_strategy="epoch",
        # save_strategy="steps",
        save_steps=10000,  # Save every 200 steps
        save_total_limit=5,  # Reduced to save space, early stopping will handle best model
        save_on_each_node=False,  # Only save on main process to avoid conflicts

        seed=args.seed,
        push_to_hub=False,

        # stable_adamw is the StableAdamW optimizer from ModernBERT:
        # https://github.com/huggingface/transformers/pull/36606/commits/7b0e57c6c220045d51de7a450d06501f807748ab
        # https://github.com/AnswerDotAI/ModernBERT/blob/main/src/optimizer.py
        optim="adamw_torch_fused",
        # optim="stable_adamw",
        # local_rank=0,  # Enable distributed training. It is
        use_mps_device=True if torch.backends.mps.is_available() else False,

        # DDP settings - Fix for unused parameters warning
        ddp_find_unused_parameters=False,
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_test_dataset['train'],
        eval_dataset=train_test_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # Stop if no improvement for 3 evals
                early_stopping_threshold=0.001  # Minimum improvement threshold
            )
        ]
    )

    # Train the model
    logger.info(f"Starting training for {dataset_name}")
    train_result = trainer.train()

    # Evaluate on test set
    logger.info(f"Evaluating {dataset_name} on test set")
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_result}")

    # Prepare results for CSV
    results = {
        'task': dataset_name,
        'model': args.model_name.split('/')[-1],
        'num_labels': len(label_names),
        **{k: v for k, v in eval_result.items() if isinstance(v, (int, float))}
    }

    # Log results
    logger.info(f"\nResults for {dataset_name}:")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


    # Parse the results to include in the final output
    results_df = pd.DataFrame([results])  # Wrap in list to create single-row DataFrame

    # Reorder columns for better readability
    base_cols = ['task', 'model', 'num_labels']
    metric_cols = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
                   'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro', 'auc_weighted']

    # Get all columns
    all_cols = base_cols + metric_cols
    other_cols = [col for col in results_df.columns if col not in all_cols]
    ordered_cols = [col for col in all_cols if col in results_df.columns] + other_cols

    results_df = results_df[ordered_cols]

    # Save results to CSV
    csv_path = os.path.join(task_output_dir, "all_tasks_performance_metrics.csv")
    results_df.to_csv(csv_path, index=False)

    logger.info(f"\nSaved performance metrics to: {csv_path}")

    # Get predictions for ROC curve logging
    logger.info("Generating predictions for ROC curve logging")
    predictions = trainer.predict(train_test_dataset['test'])

    # Extract probabilities and ground truth labels
    logits = predictions.predictions
    ground_truth = predictions.label_ids

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

    # Create label names list for wandb plotting
    class_names = [id2label[i] for i in sorted(id2label.keys())]

    logger.info(f"Class names for ROC curve: {class_names}")
    logger.info(f"Number of classes: {len(class_names)}")
    logger.info(f"Probabilities of classes: {probabilities}")
    logger.info(f"Ground truth labels: {ground_truth}")

    # Log ROC curve to wandb
    # Based on: https://docs.wandb.ai/guides/app/features/custom-charts/
    wandb.log({
        "roc_curve": wandb.plot.roc_curve(
            ground_truth,
            probabilities,  # Positive class probabilities
            labels=class_names
        )
    })

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save the model to local storage
    logger.info(f"Saving the model to {task_output_dir}...")
    model.save_pretrained(task_output_dir)

    # Save the tokenizer to local storage
    logger.info(f"Saving the tokenizer to {task_output_dir}...")
    tokenizer.save_pretrained(task_output_dir)

    # Solves warning: "destroy_process_group() was not called before program exit"
    # in case of distributed training with multiple GPUs using accelerate
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return results



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune BERT for Multiple Sentiment Analysis Tasks")

    # Data arguments
    parser.add_argument("--hf_datasets", type=str,
                        default="./../../../data/hf_datasets")

    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Proportion of data to use for testing")

    parser.add_argument('--dataset_name', type=str,
                        default="ExponentialScience/DLT-Sentiment-News",
                        help='Path to the dataset')

    # Model arguments
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base",
                        help="Pretrained model name or path")

    parser.add_argument("--tokenizer_name", type=str,
                        default=None,
                        help="Model name or path to load from")

    parser.add_argument("--model_dir", type=str,
                        default="./../../../models",
                        help="Directory to save models")

    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to load model checkpoint from")

    parser.add_argument('--num_processes', type=int,
                        default=None,
                        help='Number of parallel processes to use. Defaults to 2/3 of available CPU cores.')

    # Training arguments
    parser.add_argument("--num_epochs", type=int,
                        default=3,
                        # default=7,
                        help="Number of training epochs")

    parser.add_argument("--learning_rate", type=float,
                        default=2e-5,
                        # default=1e-5,  # To reduce the evaluation loss
                        help="Learning rate")

    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Training batch size per device")

    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Evaluation batch size per device")

    parser.add_argument("--gradient_accumulation_steps", type=int,
                        # default=1,
                        default=4,
                        help="Number of gradient accumulation steps")

    parser.add_argument("--set_per_process_memory_fraction", type=float,
                        default=0.90,
                        help="GPU memory fraction to use per process")

    parser.add_argument("--warmup_steps", type=int,
                        # default=500,
                        default=200,
                        help="Number of warmup steps")

    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Maximum sequence length (increased for title + description)")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 mixed precision training")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    # Set memory fraction if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(
            f"Setting GPU memory fraction to {args.set_per_process_memory_fraction}")
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(
                args.set_per_process_memory_fraction, device=i)

    # Load tokenizer once
    tokenizer_path = args.checkpoint_path if args.checkpoint_path else (
        args.tokenizer_name) if args.tokenizer_name else args.model_name

    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    # set model_max_length to 512 as label texts are no longer than 512 tokens
    if args.max_seq_length is not None:
        tokenizer.model_max_length = args.max_seq_length
    else:
        args.max_seq_length = tokenizer.model_max_length

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create output directory with timestamp
    date_time_run = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Store results for all tasks
    all_results = []

    if args.dataset_name == "ExponentialScience/DLT-Sentiment-News":

        logger.info(f"Experimenting with dataset: {args.dataset_name}")

        col_tasks = ["engagement_quality", "market_direction", "content_characteristics"]

        for col in col_tasks:
            logger.info(f"Training model for task: {col}")

            output_dir = os.path.join(args.model_dir, args.model_name.split("/")[-1], date_time_run, col)
            os.makedirs(output_dir, exist_ok=True)

            run_name = f"{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{col}"

            # Train the sentiment model for each task
            # No specific column labels for this dataset, so passing None
            train_model(
                dataset_name=args.dataset_name,
                args=args,
                tokenizer=tokenizer,
                task_output_dir=output_dir,
                col_labels=col,  # No specific column labels for this dataset
                run_name=run_name,
            )

            # Close previous W&B run if it exists
            if wandb.run is not None:
                logger.info(f"Closing previous W&B run for task {col}...")
                wandb.finish()

    else:
        logger.info(f"Experimenting with dataset: {args.dataset_name}")

        output_dir = os.path.join(args.model_dir, args.model_name.split("/")[-1], date_time_run)
        os.makedirs(output_dir, exist_ok=True)

        train_model(
            dataset_name=args.dataset_name,
            args=args,
            tokenizer=tokenizer,
            task_output_dir=output_dir,
            col_labels=None  # No specific column labels for this dataset
        )

    logger.info(f"\nAll models and results saved to: {output_dir}")
    logger.info("\nTraining completed successfully!")