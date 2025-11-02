import os
import argparse
import pymupdf4llm
import logging
from pathlib import Path
from glob import glob
from tqdm import tqdm
import time
import multiprocessing
import asyncio
import aiofiles
import random
from functools import partial

# Configure logging
logger = logging.getLogger(__name__)


async def write_file_async(output_file, content):
    """
    Write content to a file asynchronously.

    Args:
        output_file (str): Path to the output file.
        content (str): Content to write to the file.
    """
    async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
        await f.write(content)


def process_single_pdf(pdf_file, output_path, pdf_path):
    """
    Process a single PDF file and convert it to Markdown.

    Args:
        pdf_file (str): Path to the input PDF file.
        output_path (str): Base path for output directory.

    Returns:
        tuple: (pdf_file, time_taken, output_file, doc_md)
    """
    # Create the output path by replacing the PDF extension with .md
    # output_subdir = os.path.split(os.path.dirname(pdf_file))[-1]
    output_subdir = pdf_file.parent.relative_to(pdf_path)
    output_dir = os.path.join(output_path, output_subdir)

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.splitext(os.path.basename(pdf_file))[0] + ".md"
    output_file = os.path.join(output_dir, output_file)

    if os.path.exists(output_file):
        logger.info(f"File {output_file} already exists. Skipping...")
        return pdf_file, 0, output_file, None

    initial_time = time.time()

    try:

        # Convert the PDF to Markdown
        doc_md = pymupdf4llm.to_markdown(pdf_file, show_progress=True)

    except Exception as e:
        logger.error(f"Error processing {pdf_file}: {e}")
        logger.info(f"Removing {pdf_file} as it is not valid")
        # Remove the pdf_file
        os.remove(pdf_file)

        return pdf_file, 0, output_file, None

    final_time = time.time()
    time_taken = final_time - initial_time

    # Return the data needed for async writing
    return pdf_file, time_taken, output_file, doc_md


async def batch_process_pdfs_async(pdf_files, output_path, pdf_path, batch_size=None, num_processes=None):
    """
    Process PDFs in batches using multiprocessing with async file writing.

    Args:
        pdf_files (list): List of PDF file paths.
        output_path (str): Base path for output directory.
        batch_size (int): Size of each batch of PDFs. If None, matches num_processes.
        num_processes (int): Number of processes to use. If None, uses 2/3 of cpu_count().
    """
    if num_processes is None:
        # Use 2/3 of available CPUs, rounded down
        num_processes = max(1, int(multiprocessing.cpu_count() * 2 / 3))

    # If batch_size is not specified, set it equal to num_processes
    if batch_size is None:
        batch_size = num_processes

    # Create a partial function with the output_path parameter fixed
    process_func = partial(process_single_pdf, output_path=output_path, pdf_path=pdf_path)

    total_pdfs = len(pdf_files)
    processed = 0

    with tqdm(total=total_pdfs, desc="Converting PDFs to Markdown") as pbar:

        # Process PDFs in batches
        for i in range(0, total_pdfs, batch_size):

            # Process PDFs in batches using multiprocessing
            start_time = time.time()

            batch = pdf_files[i:i + batch_size]

            # Processing only if the batch is not empty
            if batch:

                # Process batch in parallel
                with multiprocessing.Pool(processes=num_processes) as pool:
                    for pdf_file, time_taken, output_file, doc_md in pool.imap_unordered(process_func, batch):

                        if not doc_md:
                            logger.info(f"Skipping {pdf_file} as it is already converted")
                            continue

                        # Write the file immediately
                        await write_file_async(output_file, doc_md)
                        logger.info(f"Converted and wrote {pdf_file} in {time_taken:.2f} seconds")
                        processed += 1
                        pbar.update(1)

                total_time = time.time() - start_time
                logger.info(f"Batch completed: {processed}/{total_pdfs} PDFs processed")
                logger.info(f"Total time for batch: {total_time:.2f} seconds")
                logger.info(f"Average time per PDF in Batch: {total_time / len(batch):.2f} seconds")


def batch_process_pdfs(pdf_files, output_path, pdf_path, batch_size=None, num_processes=None):
    """
    Entry point for batch processing that handles the async event loop.

    Args:
        pdf_files (list): List of PDF file paths.
        output_path (str): Base path for output directory.
        batch_size (int): Size of each batch of PDFs. If None, matches num_processes.
        num_processes (int): Number of processes to use. If None, uses 2/3 of cpu_count().
    """
    # Run the async function in the event loop
    asyncio.run(batch_process_pdfs_async(pdf_files, output_path, pdf_path, batch_size, num_processes))


def main():

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
    parser.add_argument('--batch_size', type=int,
                        default=None,
                        help='Number of PDFs to process in each batch. Defaults to match the number of processes.')
    parser.add_argument('--num_processes', type=int,
                        default=None,
                        help='Number of parallel processes to use. Defaults to 2/3 of available CPU cores.')

    args = parser.parse_args()

    # Calculate optimal process count (if not specified)
    if args.num_processes is None:
        num_processes = max(1, int(multiprocessing.cpu_count() * 2 / 3))
        logger.info(f"Using {num_processes} processes (2/3 of available {multiprocessing.cpu_count()} CPU cores)")
    else:
        num_processes = args.num_processes
        logger.info(f"Using {num_processes} processes as specified")

    # Calculate batch size if not specified
    if args.batch_size is None:
        batch_size = num_processes
        logger.info(f"Setting batch size to {batch_size} to match number of processes")
    else:
        batch_size = args.batch_size
        logger.info(f"Using batch size of {batch_size} as specified")

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
            logger.info(f"No PDF files found in {subdir_path}")
            continue

        # Avoid PDFs already converted in the output directory
        pdf_files = [pdf_file for pdf_file in pdf_files if not (output_path / pdf_file.relative_to(pdf_path)).with_suffix('.md').exists()]

        if not pdf_files:
            logger.info(f"All PDF files already converted in {subdir_path}. Skipping...")
            # If all files are already converted, exit early to save time and without any errors
            continue

        logger.info(f"Found {len(pdf_files)} PDF files to process in {subdir_path}")

        # Randomize the order of PDF files
        random.shuffle(pdf_files)

        # Process PDFs in batches using multiprocessing
        start_time = time.time()

        batch_process_pdfs(
            pdf_files,
            args.output_path,
            pdf_path,
            batch_size=batch_size,
            num_processes=num_processes
        )

        total_time = time.time() - start_time
        logger.info(f"Total time processing files in {subdir}: {total_time:.2f} seconds")
        logger.info(f"Average time per PDF in {subdir}: {total_time / len(pdf_files):.2f} seconds")


if __name__ == "__main__":
    # Resubmit the job if it fails
    for i in range(100):
        print(f"Attempt {i + 1}...")
        try:
            main()
        except Exception as e:
            print(f"Error from main logic: {e}")
            # Wait for 1 second before trying again
            time.sleep(1)
            print("Failed to run the job. Trying again...")
            continue