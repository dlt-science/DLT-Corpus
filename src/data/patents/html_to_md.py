from markitdown import MarkItDown
import glob
import os
import argparse
import asyncio
import time
import logging
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import ftfy
from pdf_to_md_batch import write_file_async

# Configure logging
logger = logging.getLogger(__name__)


def process_file(args):
    """Process a single HTML file to Markdown. This runs in a separate process."""

    html_file, output_file = args

    try:

        # Skip if output already exists
        if not output_file.exists():

            # Initialize MarkItDown in each process
            md = MarkItDown(enable_plugins=False)

            # Convert HTML to Markdown
            result = md.convert(html_file).markdown

            # Fix encoding issues
            result = ftfy.fix_text(result)

            # Clean up the result
            result = result.replace("# US Patent & Trademark Office Patent Public Search | Text View\n\n", "")
            result = result.replace("# US Patent & Trademark Office Patent Public Search | Text View", "")

            # Save the result to the output file
            with output_file.open('w', encoding='utf-8') as f:
                f.write(result)

    except Exception as e:
        logger.error(e)


async def main():

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--html_path', type=str,
                        default="./../../data/patents/htmls",
                        help='Path to the input PDF files.')
    parser.add_argument('--output_path', type=str,
                        default="./../../data/patents/markdowns",
                        help='Path to save the output Markdown files.')

    parser.add_argument('--num_processes', type=int,
                        default=None,
                        help='Number of parallel processes to use. Defaults to 2/3 of available CPU cores.')

    parser.add_argument('--batch_size', type=int,
                        default=100,
                        help='Number of files to process in each batch.')

    args = parser.parse_args()

    # Calculate optimal process count (if not specified)
    if args.num_processes is None:
        num_processes = max(5, int(multiprocessing.cpu_count() * 2 / 3))
        logger.info(f"Using {num_processes} processes (2/3 of available {multiprocessing.cpu_count()} CPU cores)")
    else:
        num_processes = args.num_processes
        logger.info(f"Using {num_processes} processes as specified")

    # Convert paths to Path objects
    html_path = Path(args.html_path)
    output_path = Path(args.output_path)

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Getting all of the HTML files in {html_path}")

    html_files = [Path(f) for f in glob.glob(os.path.join(html_path, "**/*.html"), recursive=True)]
    # md_files_processed = [Path(f) for f in glob.glob(os.path.join(output_path, "**/*.md"), recursive=True)]

    # Filter out already processed files
    html_files = [f for f in html_files if not (output_path / f.relative_to(html_path)).with_suffix('.md').exists()]

    if html_files:

        # Create list of (input, output) pairs
        file_pairs = []
        for html_file in html_files:
            output_file = output_path / html_file.relative_to(html_path).with_suffix('.md')
            if not output_file.exists():
                file_pairs.append((html_file, output_file))

        logger.info(f"Processing {len(file_pairs)} files with {num_processes} workers")

        # Calculate total number of batches
        total_batches = (len(file_pairs) + args.batch_size - 1) // args.batch_size

        # Process files in batches with overall progress
        with tqdm(total=total_batches, desc="Overall Progress", unit="batch") as batch_pbar:
            for batch_num, i in enumerate(range(0, len(file_pairs), args.batch_size), 1):
                batch = file_pairs[i:i + args.batch_size]
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")

                # Process batch in parallel
                with multiprocessing.Pool(num_processes) as pool:
                    list(tqdm(
                        pool.imap(process_file, batch),
                        total=len(batch),
                        desc=f"Batch {batch_num}/{total_batches}",
                        leave=False  # Don't leave individual batch progress bars
                    ))

        # # Process files in parallel
        # with multiprocessing.Pool(num_processes) as pool:
        #     list(tqdm(
        #         pool.imap(process_file, file_pairs),
        #         total=len(file_pairs),
        #         desc="Converting"
        #     ))

    logger.info("All files processed successfully!")

    # # Initialize MarkItDown
    # md = MarkItDown(enable_plugins=False)
    #
    # for html_file in tqdm(html_files):
    #
    #     # Create the output directory if it doesn't exist
    #     output_file = output_path / html_file.relative_to(html_path).with_suffix('.md')
    #
    #     if not os.path.exists(output_file):
    #
    #         logger.info(f"Processing file: {html_file}")
    #         result = md.convert(html_file).markdown
    #
    #         result = result.replace("# US Patent & Trademark Office Patent Public Search | Text View\n\n", "")
    #         result = result.replace("# US Patent & Trademark Office Patent Public Search | Text View", "")
    #         try:
    #             # Convert HTML to Markdown and write to file
    #             # await write_file_async(output_file, result.markdown)
    #             await write_file_async(output_file, result)
    #
    #             logger.info(f"Successfully processed {html_file} to {output_file}")
    #         except Exception as e:
    #             logger.error(f"Error processing {html_file}: {e}")
    #             continue



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