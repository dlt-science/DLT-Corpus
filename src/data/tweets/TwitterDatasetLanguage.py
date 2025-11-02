import argparse
import os
import logging
from tqdm import tqdm
import polars as pl
from ftfy import fix_text, fix_encoding
from count_tokens import tokenize_and_count_tokens
from datasets import Dataset, load_dataset, concatenate_datasets
import multiprocessing
from lingua import Language, LanguageDetectorBuilder

# Adding the path to be able to import the analytics module
import sys

sys.path.append('../../')
from models.continual_training import load_dataset_safe

# Configure logging
logger = logging.getLogger(__name__)

# Initialize lingua language detector
detector = LanguageDetectorBuilder.from_all_languages().build()


def detect_language(example):
    """
    Predict language of the given text using lingua
    """
    text = example['tweet']

    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return {'language': None}

    # Clean text for better detection
    cleaned_text = text.replace('\n', ' ').strip()

    # Detect language
    detected_language = detector.detect_language_of(cleaned_text)

    if detected_language:
        # Return ISO 639-1 language code
        lang_code = detected_language.iso_code_639_1.name.lower()
        return {'language': lang_code}
    else:
        return {'language': None}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_datasets", type=str,
                        default="./../../data/hf_datasets")

    parser.add_argument('--batch_size', type=int, default=100000,
                        help='Rows to process at once (lower = less memory)')

    parser.add_argument("--hf_dataset_repo", type=str,
                        default="ExponentialScience/Crypto-Tweets")

    args = parser.parse_args()

    # Get the number of CPUs available
    num_proc = max(1, int(2 / 3 * multiprocessing.cpu_count()))

    ds = load_dataset_safe(
        args.hf_datasets, args.hf_dataset_repo
    )

    logger.info(f"Loaded {len(ds)} tweets")

    logger.info("Going to remove non-English tweets...")

    # Predict languages in parallel
    final_dataset = ds.map(
        detect_language,
        num_proc=num_proc,
        desc="Detecting languages",
        load_from_cache_file=True
    )

    # Show example of some rows after prediction of language
    logger.info("\nExample rows after language detection:")
    for i, example in enumerate(final_dataset.select(range(min(3, len(final_dataset))))):
        logger.info(f"Row {i + 1}:")
        logger.info(f"  Date: {example['timestamp']}")
        logger.info(f"  User: {example['username']}")
        logger.info(f"  Tweet: {example['tweet'][:100]}...")
        logger.info(f"  Detected Language: {example['language']}")

    # Keep only English tweets
    final_dataset = final_dataset.filter(
        lambda x: x.get('language') == 'en',
        desc="Filtering non-English tweets",
        num_proc=num_proc,
        load_from_cache_file=True
    )
    logger.info(f"Rows after language filtering: {len(final_dataset):,}")

    # If final dataset is empty after filtering, exit
    if len(final_dataset) == 0:
        logger.warning("No English tweets found after language filtering. Exiting.")
        return

    # Display sample results after language detection
    logger.info("\nSample results after language detection:")
    sample_size = min(5, len(final_dataset))
    for i, example in enumerate(final_dataset.select(range(sample_size))):
        logger.info(f"Row {i + 1}:")
        logger.info(f"  Date: {example.get('timestamp', 'N/A')}")
        logger.info(f"  User: {example.get('username', 'N/A')}")
        logger.info(f"  Tweet: {example['tweet'][:100]}...")
        logger.info(f"  Detected Language: {example.get('language', 'None')}")

    # Save locally
    output_path = os.path.join(args.hf_datasets, args.hf_dataset_repo.split("/")[-1])
    logger.info(f"Saving to {output_path}")
    final_dataset.save_to_disk(output_path)

    # Also save as parquet (more efficient format)
    parquet_path = output_path + '.parquet'
    final_dataset.to_parquet(parquet_path)
    logger.info(f"Also saved as parquet: {parquet_path}")

    # Upload to HuggingFace Hub
    logger.info(f"Uploading to HuggingFace Hub: {args.hf_dataset_repo}")
    final_dataset.push_to_hub(args.hf_dataset_repo, private=True)
    logger.info("Upload successful!")

    logger.info("\nProcessing complete!")


if __name__ == "__main__":
    main()
