# Based on https://huggingface.co/docs/datasets/en/loading#local-and-remote-files

from datasets import load_dataset, concatenate_datasets
import logging
import os
import argparse
import multiprocessing
from tqdm import tqdm
from functools import partial


# Configure logging
logger = logging.getLogger(__name__)

def extract_filename(example, idx,
                          hf_dataset_text
                          ):

    # Get filename from the dataset
    filepath = list(hf_dataset_text.download_checksums.keys())[idx]
    filename = os.path.basename(filepath)

    return {'filename': filename}


def main():

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--md_path', type=str,
                        default="./../../data/markdown",
                        help='Path to save the output Markdown files.')

    parser.add_argument('--hf_datasets', type=str,
                        default="./../../data/hf_datasets",
                        help='Path to save the output Markdown files.')

    parser.add_argument('--num_processes', type=int,
                        default=None,
                        help='Number of parallel processes to use. Defaults to 2/3 of available CPU cores.')

    parser.add_argument("--hf_dataset_repo", type=str,
                        default="ExponentialScience/DLT-Literature")

    args = parser.parse_args()

    # Calculate optimal process count (if not specified)
    if args.num_processes is None:
        num_processes = max(1, int(multiprocessing.cpu_count() * 2 / 3))
        logger.info(f"Using {num_processes} processes (2/3 of available {multiprocessing.cpu_count()} CPU cores)")
    else:
        num_processes = args.num_processes
        logger.info(f"Using {num_processes} processes as specified")

    logger.info(f"Getting all of the subdirectories in {args.md_path}")
    subdirs = [d for d in os.listdir(args.md_path) if os.path.isdir(os.path.join(args.md_path, d))]
    logger.info(f"Found {len(subdirs)} subdirectories to process")

    # Create empty huggingface dataset
    hf_dataset = None

    for i, subdir in tqdm(enumerate(subdirs), total=len(subdirs)):

        # Only continue if the subdir is not empty of .md files
        subdir_path = os.path.join(args.md_path, subdir)

        # Get only the first .md file in the subdir
        md_files = []
        with os.scandir(subdir_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.md'):
                    md_files = [entry.name]
                    break

        if not md_files:
            logger.info(f"Skipping subdirectory '{subdir}' - no markdown files found")
            continue

        logger.info(f"Processing subdirectory '{subdir}'")

        data_files = os.path.join(subdir_path, "*.md")
        hf_dataset_subdir_dict = load_dataset("text",
                                              data_files=data_files,
                                              sample_by="document", # To avoid splitting the documents
                                              encoding="utf-8",
                                              num_proc=num_processes
                                              )

        # Extract the train split
        hf_dataset_subdir = hf_dataset_subdir_dict['train']

        logger.info(f"Found {len(hf_dataset_subdir)} documents")

        logger.info("Adding the filename as paper_id column")

        # Extract the filename and add it as a column
        hf_dataset_subdir = hf_dataset_subdir.map(
            partial(extract_filename, hf_dataset_text=hf_dataset_subdir),
            with_indices=True,
            num_proc=num_processes,
            load_from_cache_file=True,  # Use cache to save processing time
            desc="Extracting filenames"
        )

        if hf_dataset is None:
            # If hf_dataset is None, initialize it with the first subdir
            hf_dataset = hf_dataset_subdir
            # break
            continue

        hf_dataset = concatenate_datasets([hf_dataset, hf_dataset_subdir])

    if hf_dataset is None:
        raise ValueError(f"No markdown files found in '{args.md_path}'")


    # Remove the .md file extension from filename column
    hf_dataset = hf_dataset.map(lambda x: {'filename': x['filename'].replace('.md', '')},
                                num_proc=num_processes,
                                load_from_cache_file=True,
                                desc="Removing .md extension from paper_id")


    logger.info("Finished processing all subdirectories")
    logger.info(f"Total number of papers: {len(hf_dataset)}")

    # Save the Hugging Face dataset
    hf_dir = os.path.join(args.hf_datasets)
    os.makedirs(hf_dir, exist_ok=True)
    hf_dataset_path = os.path.join(hf_dir, "literature_hf")

    logger.info(f"Writing HF dataset locally to {hf_dataset_path}")
    hf_dataset.save_to_disk(hf_dataset_path)

    logger.info(f"Pushing dataset to {args.hf_dataset_repo}...")
    hf_dataset.push_to_hub(args.hf_dataset_repo, token=True, private=True)


if __name__ == "__main__":
    main()
