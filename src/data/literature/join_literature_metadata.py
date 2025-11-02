import argparse
import os
import logging
from datasets import Dataset, load_dataset
import multiprocessing
import json
import glob
from tqdm import tqdm

# Adding the path to be able to import the analytics module
import sys

sys.path.append('../../')
from models.continual_training import load_dataset_safe

# Configure logging
logger = logging.getLogger(__name__)


def normalize_record(data):
    """Normalize a single record to ensure consistent schema"""

    # Handle publicationVenue (struct/dict field)
    if data.get('publicationVenue') is None:
        data['publicationVenue'] = {
            'id': None, 'name': None, 'type': None,
            'alternate_names': None, 'issn': None,
            'url': None, 'alternate_urls': None
        }

    # Handle openAccessPdf (struct/dict field)
    if data.get('openAccessPdf') is None:
        data['openAccessPdf'] = {
            'url': None, 'status': None,
            'license': None, 'disclaimer': None
        }

    # Handle array/list fields - convert None to empty list
    list_fields = [
        'publicationTypes',
        'references',
        'fieldsOfStudy',
        's2FieldsOfStudy',
        'authors'
    ]

    for field in list_fields:
        if data.get(field) is None:
            data[field] = []

    return data

def process_json_file(json_file):
    """Process a single JSON file and return normalized JSON string"""
    try:
        with open(json_file, 'r') as infile:
            data = json.load(infile)

        # Normalize the record
        data = normalize_record(data)

        # Return as JSON string
        return json.dumps(data)
    except Exception as e:
        logger.error(f"Error processing {json_file}: {e}")
        return None


if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata_dir', type=str,
                        default="./../../data/metadata",
                        help='Path to save the output of literature metadata.')

    parser.add_argument('--literature_dir', type=str,
                        default="./../../data/metadata/literature",
                        help='Path to save the output of literature metadata.')

    parser.add_argument('--hf_datasets', type=str,
                        default="./../../data/hf_datasets",
                        help='Path to save the output Markdown files.')

    parser.add_argument("--hf_dataset_join", type=str,
                        default="ExponentialScience/DLT-Literature-PaperID")

    parser.add_argument("--hf_metadata_dataset", type=str,
                        default="ExponentialScience/DLT-Literature-Metadata")

    parser.add_argument("--hf_dataset_repo", type=str,
                        default="ExponentialScience/DLT-Literature-Final")

    args = parser.parse_args()

    # Get the number of CPUs available
    num_proc = max(1, int(2 / 3 * multiprocessing.cpu_count()))

    # Create a normalized JSONL file efficiently
    jsonl_path = os.path.join(args.metadata_dir, "normalized.jsonl")

    if not os.path.exists(jsonl_path):

        logger.info(f"Creating normalized JSONL from {args.literature_dir}")
        logger.info("This will process 40k files - may take a few minutes")

        all_json_files = glob.glob(f"{args.literature_dir}/*.json")
        logger.info(f"Found {len(all_json_files)} JSON files")

        # Filter out empty files before processing
        json_files = [f for f in all_json_files if os.path.getsize(f) > 0]
        empty_count = len(all_json_files) - len(json_files)

        if empty_count > 0:
            logger.warning(f"Skipping {empty_count} empty files (0 bytes)")

        logger.info(f"Processing {len(json_files)} non-empty files")

        # Process files in parallel and write directly to file
        processed_count = 0
        with multiprocessing.Pool(processes=num_proc) as pool:
            with open(jsonl_path, 'w') as outfile:
                with tqdm(total=len(json_files), desc="Processing JSON files", unit="files") as pbar:
                    for result in pool.imap(process_json_file, json_files, chunksize=100):
                        if result is not None:
                            outfile.write(result + '\n')
                            processed_count += 1
                        pbar.update(1)

        logger.info(f"Successfully processed and wrote {processed_count} files to {jsonl_path}")

        # with open(jsonl_path, 'w') as outfile:
        #     for idx, json_file in tqdm(enumerate(json_files), total=len(json_files)):
        #         if idx % 5000 == 0:
        #             logger.info(f"Processed {idx}/{len(json_files)} files")
        #
        #         try:
        #             with open(json_file, 'r') as infile:
        #                 data = json.load(infile)
        #
        #             # Normalize the record
        #             data = normalize_record(data)
        #
        #             # Write as single-line JSON
        #             outfile.write(json.dumps(data) + '\n')
        #         except Exception as e:
        #             logger.error(f"Error processing {json_file}: {e}")
        #             continue
        #
        # logger.info(f"Normalized JSONL created at {jsonl_path}")

        # Now load the normalized JSONL efficiently
        logger.info("Loading normalized JSONL file as dataset")
        ds_jsons = load_dataset('json', data_files=jsonl_path, split='train')
        logger.info(f"Loaded {len(ds_jsons)} records")

        # Upload to HuggingFace Hub
        logger.info(f"Uploading to HuggingFace Hub: {args.hf_metadata_dataset}")
        ds_jsons.push_to_hub(args.hf_metadata_dataset, private=True)
        logger.info("Upload successful!")

    ds_jsons = load_dataset_safe(
        args.hf_datasets, args.hf_metadata_dataset
    )

    # Load the other dataset with text to join with
    logger.info(f"Loading Huggingface dataset from {args.hf_dataset_join}")
    ds = load_dataset_safe(
        args.hf_datasets, args.hf_dataset_join
    )

    # Rename the filename column to paperId to match the metadata
    ds = ds.rename_column("filename", "paperId")

    # Get the set of paper IDs in the metadata dataset
    metadata_paper_ids = set(ds_jsons['paperId'])
    logger.info(f"Found {len(metadata_paper_ids)} unique paper IDs in the metadata dataset")

    # Get the set of paper IDs in the text dataset
    text_paper_ids = set(ds['paperId'])
    logger.info(f"Found {len(text_paper_ids)} unique paper IDs in the text dataset")

    # Find the intersection of paper IDs
    common_paper_ids = metadata_paper_ids.intersection(text_paper_ids)
    logger.info(f"Found {len(common_paper_ids)} paper IDs present in both datasets")

    # Remove duplicates from text dataset (keep first occurrence)
    logger.info("Removing duplicates from text dataset")
    seen_text = set()


    def remove_text_duplicates(example):
        if example['paperId'] in seen_text:
            return False
        seen_text.add(example['paperId'])
        return True


    ds = ds.filter(remove_text_duplicates, load_from_cache_file=False)
    logger.info(f"After deduplication: {len(ds)} text records")

    # Remove duplicates from metadata dataset (keep first occurrence)
    logger.info("Removing duplicates from metadata dataset")
    seen_meta = set()


    def remove_meta_duplicates(example):
        if example['paperId'] in seen_meta:
            return False
        seen_meta.add(example['paperId'])
        return True


    ds_jsons = ds_jsons.filter(remove_meta_duplicates, load_from_cache_file=False)
    logger.info(f"After deduplication: {len(ds_jsons)} metadata records")

    # Now filter both datasets to only include common paper IDs
    logger.info("Filtering text dataset to common paper IDs")
    ds_filtered = ds.filter(lambda x: x['paperId'] in common_paper_ids, num_proc=num_proc,
                            load_from_cache_file=True)
    logger.info(f"Filtered text dataset: {len(ds_filtered)} records")

    logger.info("Filtering metadata dataset to common paper IDs")


    def filter_metadata_papers(batch):
        return [paper_id in common_paper_ids for paper_id in batch['paperId']]


    ds_jsons_filtered = ds_jsons.filter(filter_metadata_papers, num_proc=num_proc,
                                        batched=True, load_from_cache_file=True)
    logger.info(f"Filtered metadata dataset: {len(ds_jsons_filtered)} records")

    # Sort both datasets by paperId to ensure alignment
    logger.info("Sorting datasets by paperId")
    ds_filtered = ds_filtered.sort('paperId')
    ds_jsons_filtered = ds_jsons_filtered.sort('paperId')

    # Verify they have the same length and paper IDs
    assert len(ds_filtered) == len(ds_jsons_filtered), \
        f"Dataset lengths don't match: {len(ds_filtered)} vs {len(ds_jsons_filtered)}"

    # Verify paper IDs match exactly
    assert ds_filtered['paperId'] == ds_jsons_filtered['paperId'], \
        "Paper IDs don't match after sorting!"

    logger.info(f"Successfully aligned both datasets with {len(ds_filtered)} records")

    # Join the datasets by paperId
    cols_meta = set(list(ds_jsons_filtered.column_names) + list(ds_filtered.column_names))
    logger.info("Joining datasets on paperId")
    new_data = {}
    for col in cols_meta:
        if col in ds_filtered.column_names:
            new_data[col] = ds_filtered[col]
        elif col in ds_jsons_filtered.column_names:
            new_data[col] = ds_jsons_filtered[col]
        else:
            raise ValueError(f"Column {col} not found in either dataset.")

    logger.info("Creating merged dataset")
    final_dataset = Dataset.from_dict(new_data)

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

    # Optionally clean up the JSONL file
    # os.remove(jsonl_path)