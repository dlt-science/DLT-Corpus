import os
import json
import asyncio
import aiohttp
import aiofiles
import argparse
import logging
from tqdm.asyncio import tqdm_asyncio, tqdm
import time
import dotenv
import random

# Adding the path to be able to import the analytics module
import sys

sys.path.append('../../')
from models.continual_training import load_dataset_safe

dotenv.load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


async def fetch_paper_metadata(session: aiohttp.ClientSession, paper_id: str) -> dict:
    """
    Fetch metadata for a single paper from Semantic Scholar API
    """

    headers = {
        "x-api-key": os.environ['S2_API_KEY']
    }

    # Base URL with required fields
    base_url = "https://api.semanticscholar.org/graph/v1/paper"
    fields = "url,isOpenAccess,year,authors,title,abstract,venue,publicationVenue,fieldsOfStudy,publicationTypes,publicationDate,references,s2FieldsOfStudy"

    # Construct the complete URL with paper ID and fields
    url = f"{base_url}/{paper_id}?fields={fields}"

    try:
        # Make the HTTP request
        async with session.get(url,
                               headers=headers,
                               timeout=30) as response:
            if response.status == 200:
                # Parse and return JSON response
                return await response.json()
            else:
                logger.error(f"Failed to fetch paper {paper_id}: HTTP {response.status}")

                return None

                # _counter_errors_fetch =+ 1
                # if _counter_errors_fetch <= 10:
                #     return None
                #
                # logger.info(f"Waiting 1 minute before restarting due to {_counter_errors_fetch} HTTP error...")
                #
                # await asyncio.sleep(60)  # Wait 1 minute
                # raise Exception(f"{_counter_errors_fetch} HTTP error, restarting from beginning")

    except Exception as e:
        logger.error(f"Error fetching paper {paper_id}: {str(e)}")
        return None


async def save_metadata_to_file(metadata: dict, paper_id: str, output_dir: str):
    """
    Save paper metadata to a JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename using paper ID
    filename = f"{paper_id}.json"
    filepath = os.path.join(output_dir, filename)

    # Write metadata to file asynchronously
    async with aiofiles.open(filepath, 'w') as file:
        await file.write(json.dumps(metadata, indent=2))

    logger.info(f"Saved metadata for paper {paper_id}")


# async def process_paper_id(session: aiohttp.ClientSession, paper_id: str, output_dir: str, rate_limit_lock):
#     """
#     Process a single paper ID: fetch metadata and save to file
#     """
#     # Fetch metadata from API with rate limiting
#     metadata = await fetch_paper_metadata(session, paper_id, rate_limit_lock)
#
#     if metadata:
#         # Save to file if successful
#         await save_metadata_to_file(metadata, paper_id, output_dir)
#
#     await asyncio.sleep(0.5)  # Small delay to avoid hitting rate limits too hard


async def main():
    """
    Main function to process multiple paper IDs
    """
    # # Create the rate limit lock inside main function
    # rate_limit_lock = asyncio.Lock()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata_dir', type=str,
                        default="./../../data/metadata/literature",
                        help='Path to save the output of literature metadata.')

    parser.add_argument('--hf_datasets', type=str,
                        default="./../../data/hf_datasets",
                        help='Path to save Huggingface dataset locally.')

    parser.add_argument("--hf_dataset_repo", type=str,
                        default="ExponentialScience/DLT-Literature-Final")

    args = parser.parse_args()

    os.makedirs(args.metadata_dir, exist_ok=True)

    # Load dataset to get paper IDs
    ds = load_dataset_safe(
        args.hf_datasets, args.hf_dataset_repo
    )

    paper_ids = list(ds['filename'])

    # # Filter out already downloaded files
    # downloaded_files = [os.path.basename(f) for f in os.listdir(args.metadata_dir)]
    # paper_ids = [pid for pid in paper_ids if f"{pid}.json" not in downloaded_files]

    # Filter out already downloaded files - optimized version
    downloaded_files = {os.path.splitext(f)[0] for f in os.listdir(args.metadata_dir) if f.endswith('.json')}
    paper_ids = [pid for pid in paper_ids if pid not in downloaded_files]

    # Shuffle the paper IDs to distribute load
    random.shuffle(paper_ids)

    logger.info(f"Processing {len(paper_ids)} paper IDs...")

    # Create one session for all requests
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        for paper_id in tqdm(paper_ids):
            metadata = await fetch_paper_metadata(session, paper_id)

            if metadata:
                await save_metadata_to_file(metadata, paper_id, args.metadata_dir)

            time.sleep(1)  # Wait between requests

    # # Create HTTP session with SSL disabled for compatibility
    # async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
    #     # Create tasks for all paper IDs
    #     tasks = [
    #         process_paper_id(session, paper_id, args.metadata_dir, rate_limit_lock)
    #         for paper_id in paper_ids
    #     ]
    #
    #     # Execute all tasks with progress bar using tqdm_asyncio
    #     await tqdm_asyncio.gather(*tasks, desc="Downloading metadata")

    logger.info(f"Completed processing {len(paper_ids)} paper IDs")


if __name__ == "__main__":
    # Resubmit the job if it fails
    for i in range(100):
        logger.info(f"Attempt {i + 1}...")
        try:
            # Run the async main function
            asyncio.run(main())
            break
        except Exception as e:
            logger.info(f"Error from main logic: {e}")
            # Wait for 5 second before trying again
            time.sleep(5)
            logger.info("Failed to run the job. Trying again...")
            continue
