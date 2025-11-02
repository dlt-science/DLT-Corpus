import os
import argparse
import logging
import random
from glob import glob
from tqdm import tqdm
import time
from pathlib import Path
from pdf_to_md_batch import process_single_pdf, write_file_async
import asyncio


# Configure logging
logger = logging.getLogger(__name__)

async def main():

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_path', type=str,
                        default="./../../data/semanticscholar/open_access_pdfs",
                        help='Path to the input PDF files.')
    parser.add_argument('--output_path', type=str,
                        default="./../../data/markdown",
                        help='Path to save the output Markdown files.')
    args = parser.parse_args()

    # Convert paths to Path objects
    pdf_path = Path(args.pdf_path)
    output_path = Path(args.output_path)

    logger.info(f"Getting all of the subdirectories in {pdf_path}")
    subdirs = [d for d in os.listdir(pdf_path) if os.path.isdir(os.path.join(pdf_path, d))]
    logger.info(f"Found {len(subdirs)} subdirectories to process")

    # Collect PDF files
    pattern = '**/*.pdf'

    # Shuffle the subdirs to randomize processing order
    random.shuffle(subdirs)

    for subdir in tqdm(subdirs):
        subdir_path = pdf_path / subdir
        pdf_files = [Path(p) for p in glob(str(subdir_path / pattern), recursive=True)]

        if not pdf_files:
            logger.info(f"No PDF files found in {pdf_path}")
            continue

        # Avoid PDFs already converted in the output directory
        pdf_files = [pdf_file for pdf_file in pdf_files if
                     not (output_path / pdf_file.relative_to(pdf_path)).with_suffix('.md').exists()]

        if not pdf_files:
            logger.info(f"All PDF files already converted in {subdir_path}. Skipping...")
            # If all files are already converted, exit early to save time and without any errors
            continue

        # Randomize the order of PDF files
        random.shuffle(pdf_files)

        logger.info(f"Found {len(pdf_files)} PDF files to process in {subdir_path}")

        # Process PDFs in batches using multiprocessing
        start_time = time.time()

        for pdf_file in tqdm(pdf_files):

            pdf_file, time_taken, output_file, doc_md = process_single_pdf(pdf_file, output_path=output_path,
                                                                           pdf_path=pdf_path)

            if not doc_md:
                logger.info(f"Skipping {pdf_file} as it is already converted")
                continue

            await write_file_async(output_file, doc_md)

            total_time = time.time() - start_time
            logger.info(f"Total time processing {pdf_file}: {total_time:.2f} seconds")


if __name__ == "__main__":
    # Resubmit the job if it fails
    for i in range(100):
        print(f"Attempt {i + 1}...")
        try:
            asyncio.run(main())
        except Exception as e:
            print(f"Error from main logic: {e}")
            # Wait for 1 second before trying again
            time.sleep(1)
            print("Failed to run the job. Trying again...")
            continue

