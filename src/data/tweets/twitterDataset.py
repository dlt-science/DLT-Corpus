import argparse
import os
import logging
import glob
from tqdm import tqdm
import polars as pl
from ftfy import fix_text, fix_encoding
from datasets import Dataset, load_dataset, concatenate_datasets
import multiprocessing
import gc

# Configure logging
logger = logging.getLogger(__name__)


def get_csv_columns(csv_file):
    """
    Read only first row of CSV to get column names without loading entire file
    """
    try:
        # scan_csv creates lazy dataframe - doesn't load data yet
        df = pl.scan_csv(csv_file, n_rows=1)
        return df.columns
    except Exception as e:
        logger.error(f"Error reading {csv_file}: {e}")
        return []


def standardize_column_names(columns):
    """
    Map various column names to standard names: date, username, tweet
    """
    # Define possible variations for each standard column
    date_variants = ['created_at', 'date', 'time', 'timestamp', 'datetime', 'date_time']
    user_variants = ['username', 'user_name', 'user', 'screen_name', 'author']
    tweet_variants = ['tweet', 'text', 'content', 'full_text', 'message', 'tweet_text']

    # Create mapping from found columns to standard names
    column_map = {}

    # Convert all columns to lowercase for matching
    columns_lower = [col.lower() for col in columns]

    variants = {
        'timestamp': date_variants,
        'username': user_variants,
        'tweet': tweet_variants
    }

    for col_name, col_variants in variants.items():
        for variant in col_variants:
            if variant in columns_lower:
                idx = columns_lower.index(variant)
                column_map[columns[idx]] = col_name
                break

    return column_map


def clean_tweet_text(example):
    """
    Basic text cleaning for tweets
    """
    if example['tweet']:
        # Remove extra whitespace between words and trim
        example['tweet'] = ' '.join(str(example['tweet']).split()).strip()

        # Fix encoding issues
        example['tweet'] = fix_text(example['tweet'])
        example['tweet'] = fix_encoding(example['tweet'])

    return example


def process_csv_file(csv_file, batch_size=50000):
    """
    Process a single CSV file in batches and return list of datasets
    """
    datasets = []

    # Get column names
    columns = get_csv_columns(csv_file)
    if not columns:
        return datasets

    # Find which columns we need
    column_map = standardize_column_names(columns)
    if not column_map:
        logger.warning(f"No relevant columns in {csv_file}")
        return datasets

    logger.info(f"Processing {os.path.basename(csv_file)}")
    logger.info(f"  Found columns: {list(column_map.keys())} -> {list(column_map.values())}")

    try:
        # Create batched reader for memory efficiency
        reader = pl.read_csv_batched(
            csv_file,
            batch_size=batch_size,
            ignore_errors=True,  # Skip bad rows
            # low_memory=True  # Use less memory
        )

        # Process each batch
        batch_num = 0
        while True:
            # Read next batch
            batch_df = reader.next_batches(1)
            if not batch_df:
                break

            batch_df = batch_df[0]  # Get first (and only) batch
            batch_num += 1

            # Select only columns we need
            cols_to_select = [col for col in column_map.keys() if col in batch_df.columns]
            batch_df = batch_df.select(cols_to_select)

            # Handle special case: merge date and time if both exist
            if 'date' in batch_df.columns and 'time' in batch_df.columns:
                logger.info("Found the date and time columns - combining into timestamp")
                # Combine date and time columns into a single datetime string
                batch_df = batch_df.with_columns(
                    (pl.col('date').cast(pl.Utf8) + ' ' + pl.col('time').cast(pl.Utf8)).alias('timestamp')
                )
                # Drop the separate date_part and time_part columns
                batch_df = batch_df.drop(['date', 'time'])

            elif 'date' in batch_df.columns and 'datetime' in batch_df.columns:
                logger.info("Found the date and datetime columns - using datetime as timestamp")
                # If both date and datetime exist, prefer datetime as timestamp
                batch_df = batch_df.rename({'datetime': 'timestamp'})

            # Rename columns to standard names
            for old_name, new_name in column_map.items():
                if old_name in batch_df.columns:
                    batch_df = batch_df.rename({old_name: new_name})

            # Add missing columns as None
            for col in ['timestamp', 'username', 'tweet']:
                if col not in batch_df.columns:
                    batch_df = batch_df.with_columns(pl.lit(None).alias(col))

            # Select final columns in correct order
            batch_df = batch_df.select(['timestamp', 'username', 'tweet'])

            # Convert to HuggingFace Dataset
            dataset = Dataset.from_polars(batch_df)
            datasets.append(dataset)

            # Clean memory
            del batch_df
            gc.collect()

    except Exception as e:
        logger.error(f"Error processing {csv_file}: {e}")

    logger.info(f"Finished processing {csv_file}")

    # Show example row in the last processed batch
    if dataset:
        example = dataset[0]
        logger.info(f"Example row: Date: {example['timestamp']}, User: {example['username']}, Tweet: {example['tweet'][:100]}...")

    return datasets

def process_json_files(json_dir):

    cols_to_keep = ['timestamp', 'username', 'tweet']
    if json_dir:
        try:
            logger.info(f"\nProcessing JSON files from {json_dir}")
            ds_jsons = load_dataset(json_dir)

            # Standardize column names
            ds_jsons = ds_jsons.rename_column('created_at', 'timestamp')

            # Keep only columns we need
            cols_to_remove = [col for col in ds_jsons.column_names
                              if col not in cols_to_keep]

            if cols_to_remove:
                ds_jsons = ds_jsons.remove_columns(cols_to_remove)

            # Add missing columns
            for col in cols_to_keep:
                if col not in ds_jsons.column_names:
                    dataset = dataset.add_column(col, [None] * len(dataset))

            return ds_jsons

        except Exception as e:
            logger.error(f"Error loading JSON files: {e}")
            return None

def main():

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str,
                        default="./../../data/Tweets",
                        help="Path to the directory containing the dataset")

    parser.add_argument("--hf_datasets", type=str,
                        default="./../../data/hf_datasets")

    parser.add_argument("--json_dir", type=str,
                        default="./../../data/Tweets/CrypTop12",
                        help="Path to the directory containing the dataset")

    parser.add_argument('--batch_size', type=int, default=100000,
                        help='Rows to process at once (lower = less memory)')

    parser.add_argument("--hf_dataset_repo", type=str,
                        default="ExponentialScience/Crypto-Tweets")

    args = parser.parse_args()

    # Get the number of CPUs available
    num_proc = 1/2 * multiprocessing.cpu_count()

    # Find all CSV files
    csv_files = glob.glob(os.path.join(args.dataset_dir, "**/*.csv"), recursive=True)
    logger.info(f"Found {len(csv_files)} files in {args.json_dir}")

    # Process each CSV file
    all_datasets = []
    for csv_file in tqdm(csv_files):
        # Get file size in GB for logging
        size_mb = os.path.getsize(csv_file) / (1024 * 1024 * 1024)
        logger.info(f"\nProcessing {csv_file} ({size_mb:.1f} GB)")

        columns = get_csv_columns(csv_file)
        logger.info(f"  Columns: {columns}")

        # Process this CSV file
        csv_datasets = process_csv_file(csv_file, args.batch_size)
        all_datasets.extend(csv_datasets)

    # Process JSON files if directory provided
    ds_jsons = process_json_files(args.json_dir)

    if ds_jsons:
        all_datasets.append(ds_jsons)

    # Check if we have any data
    if not all_datasets:
        logger.error("No data was loaded!")
        return

    # Combine all datasets into one
    logger.info(f"\nCombining {len(all_datasets)} datasets...")
    final_dataset = concatenate_datasets(all_datasets)
    logger.info(f"Total rows: {len(final_dataset):,}")

    # Clean tweet text
    logger.info("Cleaning tweet text...")
    final_dataset = final_dataset.map(clean_tweet_text, desc="Cleaning tweets",
                                      num_proc=int(num_proc),
                                     load_from_cache_file=True)

    # Remove rows with empty tweets
    logger.info("Removing empty tweets...")
    final_dataset = final_dataset.filter(
        lambda x: x['tweet'] is not None and len(str(x['tweet']).strip()) > 0,
        desc="Filtering empty tweets"
    )
    logger.info(f"Rows after filtering: {len(final_dataset):,}")

    logger.info(f"Going to remove duplicate tweets...")

    # Convert to Polars DataFrame for efficient deduplication
    df = pl.from_arrow(final_dataset.data.table)
    initial_count = len(df)
    df = df.unique(subset=['tweet'], keep='first')
    final_dataset = Dataset.from_polars(df)
    logger.info(
        f"Removed {initial_count - len(df):,} duplicate tweets. Rows after deduplication: {len(final_dataset):,}")

    logger.info("\nProcessing complete!")

if __name__ == "__main__":

    main()
