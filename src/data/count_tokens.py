import logging
from functools import partial
from datasets import Dataset
from transformers import AutoTokenizer
import argparse
import multiprocessing
from typing import Dict, Any

# Adding the path to be able to import the analytics module
import sys

sys.path.append('./../')
from models.continual_training import load_dataset_safe

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_function(examples: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Tokenize the text and return token counts.

    Args:
        examples: Batch of examples containing 'text' column
        tokenizer: Hugging Face tokenizer

    Returns:
        Dictionary with 'token_count' for each example
    """
    # Tokenize the text
    tokenized = tokenizer(examples["text"], truncation=False, padding=False)

    # Count tokens for each example in the batch
    token_counts = [len(tokens) for tokens in tokenized["input_ids"]]

    return {"total_tokens": token_counts}


def tokenize_and_count_tokens(
        dataset: Dataset,
        tokenizer,
        num_processes: int = 4,
        batch_size: int = 1000
) -> Dataset:
    """
    Tokenize dataset and add token count column.

    Args:
        dataset: Hugging Face dataset
        tokenizer: Tokenizer to use
        num_processes: Number of processes for multiprocessing
        batch_size: Batch size for processing

    Returns:
        Dataset with added 'token_count' column
    """
    # Create preprocessing function with tokenizer
    preprocess_fn = partial(preprocess_function, tokenizer=tokenizer)

    logger.info(f"Tokenizing the dataset with {num_processes} processes")
    logger.info(f"Dataset size: {len(dataset):,} examples")

    # Apply tokenization and count tokens
    processed_dataset = dataset.map(
        preprocess_fn,
        num_proc=num_processes,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=True,  # Use cache to save processing time
        desc="Tokenizing and counting tokens"
    )

    # Calculate total tokens
    total_tokens = sum(processed_dataset["total_tokens"])
    logger.info(f"Total tokens counted: {total_tokens:,}")

    return processed_dataset


def main():
    parser = argparse.ArgumentParser(description="Tokenize Hugging Face dataset and count tokens")

    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the Hugging Face dataset to process")


    parser.add_argument("--tokenizer_name", type=str, default="answerdotai/ModernBERT-base",
                        help="Name of the tokenizer to use (default: answerdotai/ModernBERT-base)")

    parser.add_argument("--hf_datasets", type=str,
                        default="./../../../data/hf_datasets")

    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of processes for multiprocessing (default: 4)")

    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for processing (default: 1000)")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the processed dataset to Hugging Face Hub")

    parser.add_argument("--hub_repo_name", type=str, default=None,
                        help="Repository name for pushing to Hub (required if --push_to_hub is used)")

    args = parser.parse_args()

    if args.num_processes is None:
        # Use 2/3 of available CPUs, rounded down
        args.num_processes = max(4, int(multiprocessing.cpu_count() * 2 / 3))

    try:
        ds = load_dataset_safe(
            args.hf_datasets, args.dataset_name
        )

        logger.info(f"Dataset loaded successfully. Shape: {ds}")
        logger.info(f"Columns: {ds.column_names}")

        columns = ds.column_names

        # Check if 'text' column exists
        if "text" not in columns and "code" not in columns and "tweet" not in columns:
            logger.error("Dataset must contain a 'text', 'code', and 'tweet' column")
            raise ValueError("Dataset must contain a 'text' or 'code' column")

        original_text_col_name = None
        if "tweet" in columns:
            original_text_col_name = "tweet"
            logger.info(f"Renaming 'tweet' column to 'text'")
            ds = ds.rename_column("tweet", "text")

        # Check if 'total_tokens' column already exists
        if "total_tokens" in columns:
            logger.info("'total_tokens' column already exists in the dataset")
            total_tokens = sum(ds["total_tokens"])
            avg_tokens = total_tokens / len(ds)
            logger.info(f"Sum of existing token counts: {total_tokens:,}")

            logger.info(f"Total examples: {len(ds):,}")
            logger.info(f"Total tokens: {total_tokens:,}")
            logger.info(f"Average tokens per example: {avg_tokens:.2f}")

            # Finish processing if already counted
            logger.info("Exiting without reprocessing.")
            return

        # Load tokenizer
        logger.info(f"Loading tokenizer: {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize and count tokens
        processed_dataset = tokenize_and_count_tokens(
            dataset=ds,
            tokenizer=tokenizer,
            num_processes=args.num_processes,
            batch_size=args.batch_size
        )

        # Push to hub if requested
        if args.hub_repo_name is not None:

            # Rename text column back if it was renamed
            if original_text_col_name is not None:
                logger.info(f"Renaming 'text' column back to '{original_text_col_name}'")
                processed_dataset = processed_dataset.rename_column("text",
                                                                    original_text_col_name)

            logger.info(f"Pushing processed dataset to: {args.hub_repo_name}")
            processed_dataset.push_to_hub(args.hub_repo_name)
            logger.info("Dataset pushed successfully!")

        # Final summary
        total_tokens = sum(processed_dataset["total_tokens"])
        avg_tokens = total_tokens / len(processed_dataset)
        logger.info(f"Processing complete!")
        logger.info(f"Total examples: {len(processed_dataset):,}")
        logger.info(f"Total tokens: {total_tokens:,}")
        logger.info(f"Average tokens per example: {avg_tokens:.2f}")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()