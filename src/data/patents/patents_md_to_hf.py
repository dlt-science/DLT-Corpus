from datasets import load_dataset, Dataset, Features, Value
import logging
import os
import argparse
import multiprocessing
import pandas as pd
from functools import partial

# Configure logging
logger = logging.getLogger(__name__)


# Add metadata columns to the filtered text dataset
def add_metadata(example, patent_to_metadata_idx, metadata_dataset, cols_meta, col_name='Patent Number'):
    patent_num = example[col_name]
    if patent_num in patent_to_metadata_idx:
        metadata_idx = patent_to_metadata_idx[patent_num]
        metadata_row = metadata_dataset[metadata_idx]

        # Add all metadata columns to the example
        for col in cols_meta:
            if col != col_name:  # Skip col_name as it's already there
                # example[col] = metadata_row[col]
                value = metadata_row[col]

                # Handle None/null values by converting to empty string
                example[col] = value if value is not None else ""

    return example


def extract_patent_number(example, idx,
                          hf_dataset_text
                          ):

    # Get filename from the dataset
    filepath = list(hf_dataset_text.download_checksums.keys())[idx]
    filename = os.path.basename(filepath)

    # Remove .md extension
    patent_number = filename.replace('.md', '')

    return {'Patent Number': patent_number}


def hf_inner_join(hf_dataset_text, hf_dataset_with_patent, metadata_dataset, num_processes, cols_meta, col_name):

    # Create a set of patent numbers from metadata for faster lookup
    metadata_patent_numbers = set(metadata_dataset[col_name])
    text_patent_numbers = set(hf_dataset_with_patent[col_name])

    filtered_text_dataset = hf_dataset_with_patent.filter(
        lambda x: x[col_name] in metadata_patent_numbers,
        num_proc=num_processes
    )
    filtered_metadata_dataset = metadata_dataset.filter(
        lambda x: x[col_name] in text_patent_numbers,
        num_proc=num_processes
    )

    logger.info(
        f"After filtering: {len(filtered_text_dataset)} matching text records (dropped {len(hf_dataset_text) - len(filtered_text_dataset)} non-matching)")
    logger.info(
        f"After filtering: {len(filtered_metadata_dataset)} matching metadata records (dropped {len(metadata_dataset) - len(filtered_metadata_dataset)} non-matching)")

    # Mapping from patent number to metadata index
    patent_to_metadata_idx = {pn: idx for idx, pn in enumerate(filtered_metadata_dataset[col_name])}

    logger.info("Merging with metadata...")

    # Partial function to add metadata passing more arguments than just the example
    add_metadata_func = partial(add_metadata, patent_to_metadata_idx=patent_to_metadata_idx,
                                metadata_dataset=filtered_metadata_dataset, cols_meta=cols_meta)

    # Merge the filtered text dataset with the filtered metadata
    hf_dataset = filtered_text_dataset.map(
        add_metadata_func,
        num_proc=num_processes,
    )

    return hf_dataset


def main():

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--patents_path', type=str,
                        default="./../../data/patents",
                        help='Path to save the output of patents text.')

    parser.add_argument('--md_path', type=str,
                        default="./../../data/patents/markdowns",
                        help='Path to save the output Markdown files.')

    parser.add_argument('--hf_datasets', type=str,
                        default="./../../data/hf_datasets",
                        help='Path to save Huggingface dataset locally.')

    parser.add_argument('--num_processes', type=int,
                        default=None,
                        help='Number of parallel processes to use. Defaults to 2/3 of available CPU cores.')

    parser.add_argument("--hf_dataset_repo", type=str,
                        default="ExponentialScience/DLT-Patents")

    args = parser.parse_args()

    metadata_dir = os.path.join(args.patents_path, "metadata")

    cols_meta = ["Document ID", "Date Published", "Family ID", "Title", "CPCI", "CPCA","Inventor","Assignee",
                 "Application Number","Filing Date", "Primary Examiner", "Assistant Examiner",
                 "OR", "XREF", "Applicant Name", "Notes", "Notes/Tagged", "Relevancy", "Database", "Patent Number"]

    metadata_df = pd.read_csv(os.path.join(metadata_dir, "metadata.csv"), usecols=cols_meta)
    # Drop rows with null Patent Number in metadata
    metadata_df = metadata_df.dropna(subset=['Patent Number'])

    logger.info(f"Loaded metadata with {len(metadata_df)} records")

    # Convert metadata to HuggingFace dataset
    metadata_dataset = Dataset.from_pandas(metadata_df)

    # Calculate optimal process count (if not specified)
    if args.num_processes is None:
        num_processes = max(5, int(multiprocessing.cpu_count() * 2 / 3))
        logger.info(f"Using {num_processes} processes (2/3 of available {multiprocessing.cpu_count()} CPU cores)")
    else:
        num_processes = args.num_processes
        logger.info(f"Using {num_processes} processes as specified")

    data_files = os.path.join(args.md_path, "*.md")

    # Define features
    features = Features({
        'text': Value('string'),
    })

    logger.info(f"Loading all .md files in {args.md_path}")

    hf_dataset_text = load_dataset("text",
                                     data_files=data_files,
                                     features=features,
                                     sample_by="document",  # To avoid splitting the documents
                                     encoding="utf-8",
                                     num_proc=num_processes
                                     )['train']

    # Extract patent numbers from filenames
    logger.info("Extracting patent numbers from filenames...")

    # Create a partial function
    extract_patent_number_func = partial(extract_patent_number, hf_dataset_text=hf_dataset_text)

    # Add Patent Number column using map
    hf_dataset_with_patent = hf_dataset_text.map(
        extract_patent_number_func,
        with_indices=True,
        num_proc=num_processes
    )

    # Drop any rows where Patent Number might be None/null (shouldn't happen with file names, but good practice)
    hf_dataset_with_patent = hf_dataset_with_patent.filter(
        lambda x: x['Patent Number'] is not None and x['Patent Number'] != '',
        num_proc=num_processes
    )
    logger.info(f"Text dataset has {len(hf_dataset_with_patent)} records after ensuring non-null Patent Numbers")


    hf_dataset = hf_inner_join(hf_dataset_text, hf_dataset_with_patent, metadata_dataset, num_processes, cols_meta, col_name='Patent Number')

    # Save the Hugging Face dataset
    hf_dir = os.path.join(args.hf_datasets)
    os.makedirs(hf_dir, exist_ok=True)
    hf_dataset_path = os.path.join(hf_dir, "patents_hf")

    logger.info(f"Writing HF dataset locally to {hf_dataset_path}")
    hf_dataset.save_to_disk(hf_dataset_path)

    logger.info(f"Pushing dataset to {args.hf_dataset_repo}...")
    hf_dataset.push_to_hub(args.hf_dataset_repo, token=True, private=True)


if __name__ == "__main__":
    main()