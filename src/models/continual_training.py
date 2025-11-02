import argparse
import logging
import os
import glob

import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from transformers import (DataCollatorForLanguageModeling, AutoTokenizer,
                          AutoModelForMaskedLM, TrainingArguments, Trainer, set_seed, logging as transformers_logging)
from accelerate import Accelerator

from functools import partial
from huggingface_hub import login
from multiprocessing import cpu_count
import pandas as pd

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# os.environ["PYTHONUNBUFFERED"] = "1"

# Configure logging
logger = logging.getLogger(__name__)

# To get logging of trainer
transformers_logging.enable_progress_bar()
transformers_logging.set_verbosity_info()

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add debugging env
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Optimize for multi-core CPU usage
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads per process
os.environ["MKL_NUM_THREADS"] = "4"  # Limit MKL threads per process
torch.set_num_threads(4)  # Limit PyTorch threads per process

os.getenv("WANDB_PROJECT")  # name W&B project

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} CUDA GPUs")
    if num_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_dataset_safe(hf_datasets_path: str, dataset_name: str) -> Dataset:
    """Load dataset from local cache or download from hub"""
    file_path = os.path.join(hf_datasets_path, dataset_name)

    if os.path.exists(file_path):
        logger.info(f"Loading dataset from local path: {file_path}")
        dataset = load_from_disk(file_path)
    else:
        logger.info(f"Loading dataset from Hugging Face hub: {dataset_name}")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(file_path)

    # Get the dataset without the "train" split
    if "train" in dataset:
        logger.info("Using 'train' split from the dataset")
        dataset = dataset["train"]

    # df = dataset_dict["train"].to_pandas()
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    return dataset


def extract_text_column(dataset, dataset_name):
    """
    Extract and standardize the text column from a dataset.
    Only keeps the 'text' column, removes all others.

    Args:
        dataset: HuggingFace dataset
        dataset_name: Name of the dataset (for logging)

    Returns:
        Dataset with only 'text' column, or None if no suitable text column found
    """
    current_columns = dataset.column_names
    logger.info(f"Dataset {dataset_name} has columns: {current_columns}")

    if "tweet" in current_columns:
        logger.info(f"Renaming 'tweet' column to 'text'")
        dataset = dataset.rename_column("tweet", "text")

    # Check if 'text' column already exists
    if 'text' in current_columns:
        logger.info(f"Dataset {dataset_name} already has 'text' column")
        # Remove all columns except 'text'
        columns_to_remove = [col for col in current_columns if col != 'text']
        if columns_to_remove:
            logger.info(f"Removing columns {columns_to_remove} from dataset {dataset_name}")
            dataset = dataset.remove_columns(columns_to_remove)
        return dataset


def load_and_combine_datasets(hf_datasets_dir, dataset_names):
    """
    Load multiple datasets and combine them into a single dataset.
    Only keeps the 'text' column, ignoring all other columns.

    Args:
        hf_datasets_dir: Directory for HuggingFace datasets cache
        dataset_names: List of dataset names to load and combine

    Returns:
        Combined dataset with only 'text' column
    """
    datasets_to_combine = []

    for dataset_name in dataset_names:
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            # Try loading with load_dataset_safe first
            ds = load_dataset_safe(hf_datasets_dir, dataset_name)
            logger.info(f"Successfully loaded {dataset_name} with {len(ds)} samples")

            # Process dataset to extract only text column
            processed_ds = extract_text_column(ds, dataset_name)
            if processed_ds is not None:
                datasets_to_combine.append(processed_ds)

        except Exception as e:
            logger.warning(f"Failed to load {dataset_name} with load_dataset_safe: {e}")
            try:
                # Fallback to standard load_dataset
                ds = load_dataset(dataset_name, split="train", cache_dir=hf_datasets_dir)
                logger.info(f"Successfully loaded {dataset_name} with load_dataset, {len(ds)} samples")

                # Process dataset to extract only text column
                processed_ds = extract_text_column(ds, dataset_name)
                if processed_ds is not None:
                    datasets_to_combine.append(processed_ds)

            except Exception as e2:
                logger.error(f"Failed to load {dataset_name} with both methods: {e2}")
                continue

    if not datasets_to_combine:
        raise ValueError("No datasets could be loaded successfully")

    if len(datasets_to_combine) == 1:
        logger.info("Only one dataset loaded, returning as-is")
        return datasets_to_combine[0]

    logger.info(f"Combining {len(datasets_to_combine)} datasets...")

    # All datasets have only 'text' column, so we can safely combine them
    combined_dataset = concatenate_datasets(datasets_to_combine)

    total_samples = len(combined_dataset)
    logger.info(f"Successfully combined datasets. Total samples: {total_samples}")

    # Log the distribution of samples from each dataset
    sample_count = 0
    for i, ds in enumerate(datasets_to_combine):
        dataset_samples = len(ds)
        percentage = (dataset_samples / total_samples) * 100
        logger.info(f"Dataset {dataset_names[i]}: {dataset_samples} samples ({percentage:.1f}%)")
        sample_count += dataset_samples

    # Shuffle the combined dataset if there are multiple datasets
    if len(dataset_names) > 1:
        logger.info("Shuffling the combined dataset to mix samples from different sources")
        combined_dataset = combined_dataset.shuffle(seed=42)

    return combined_dataset


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    if not os.path.exists(output_dir):
        logger.info(f"No checkpoint found in {output_dir}")
        return None

    # Look for checkpoint directories
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        return None

    # Extract step numbers and find the latest one
    checkpoint_steps = []
    for checkpoint in checkpoints:
        try:
            step = int(os.path.basename(checkpoint).split("-")[1])
            checkpoint_steps.append((step, checkpoint))
        except (IndexError, ValueError):
            continue

    if not checkpoint_steps:
        logger.info(f"No checkpoint found in {output_dir}")
        return None

    # Return the checkpoint with the highest step number
    latest_checkpoint = max(checkpoint_steps, key=lambda x: x[0])[1]
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported
    # it instead of this drop, you can customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

def gen_lm_dataset(ds, tokenizer, block_size):
    # Determine the number of processes for map using 2/3 of available CPUs
    num_processes_for_map = int(cpu_count() * 2 / 3)
    logger.info(f"Using {num_processes_for_map} CPUs for processes for dataset mapping")

    # Pass multiple arguments to the preprocess function
    preprocesses_fn = partial(preprocess_function,
                              tokenizer=tokenizer)

    logger.info(f"Tokenizing the dataset with {num_processes_for_map} processes")
    tok_ds = ds.map(preprocesses_fn,
                    num_proc=num_processes_for_map,
                    batched=True,
                    remove_columns=list(ds.column_names),
                    load_from_cache_file=True,  # Use cache to save processing time
                    )

    group_texts_fn = partial(group_texts,
                             block_size=block_size)

    logger.info(f"Grouping texts into blocks of size {block_size}")
    lm_dataset = tok_ds.map(group_texts_fn,
                            batched=True,
                            num_proc=num_processes_for_map,  # Number of CPUs
                            load_from_cache_file=True,
                            )
    return lm_dataset

def train(model, tokenizer, data_collator, train_test_dataset, output_dir, args,
          set_per_process_memory_fraction=0.90):
    # Initialize accelerator
    accelerator = Accelerator()

    # Check if multiple GPUs are available
    # num_gpus = torch.cuda.device_count()
    num_gpus = accelerator.num_processes

    logger.info(f"Using {num_gpus} GPUs for training")

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        logger.info(f"Setting up a hard limit of {set_per_process_memory_fraction * 100}% "
                  f"of GPU memory per process that PyTorch can allocate to prevent PyTorch "
                  f"from using 100% of GPU memory")

        # Set memory fraction to prevent OOM
        for i in range(num_gpus):
            torch.cuda.set_per_process_memory_fraction(set_per_process_memory_fraction, device=i)

    # Create logs directory explicitly
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger.info(f"Tensorboard logs will be written to: {logs_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logs_dir,  # Explicit logs directory
        eval_strategy="steps",
        eval_steps=5000,  # Evaluate every 5000 steps

        learning_rate=5e-5,  # More conservative

        # learning_rate=2e-5, # Used for BERT-based and SciBERT-based models
        # learning_rate=3e-4,  # keep same as modernbert
        lr_scheduler_type="linear",
        num_train_epochs=args.num_train_epochs,

        # To handle out-of-memory errors by loading 2 batches per GPU of 20 GB
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        # #
        # gradient_accumulation_steps=2 if num_gpus > 1 else 4,

        # Values based on: https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/train_st.py
        # per_device_train_batch_size=2,
        # per_device_eval_batch_size=2,
        # warmup_ratio=0.05,
        warmup_ratio=0.1, # default value
        weight_decay=0.01,

        # Using similar Weight Decay than ModernBERT: https://github.com/AnswerDotAI/ModernBERT/blob/main/main.py
        # weight_decay=1e-5,
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency

        # periodically move predictions to CPU and free GPU memory
        # eval_accumulation_steps=2,
        eval_on_start=True,

        # Mixed precision training
        bf16=True,  # Use bfloat16 for better memory efficiency
        bf16_full_eval=True,

        # Checkpoint management
        save_strategy="steps",  # Save checkpoints periodically
        save_steps=1000,  # Save every 1000 steps
        save_total_limit=10,  # Keep only 10 checkpoints to save disk space

        # Logging
        logging_steps=10,
        logging_strategy="steps",
        # report_to="tensorboard",
        run_name=pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S") if args.run_name is None else args.run_name,
        report_to="wandb",  # enable logging to W&B
        logging_first_step=True,  # Log the first step
        # log_level="info",

        # Memory management
        remove_unused_columns=False,
        dataloader_pin_memory=True,  # Enable pin memory with sufficient RAM
        # dataloader_num_workers=min(8, os.cpu_count() // 2),  # Use 8 workers minimum or half of available CPU cores
        dataloader_num_workers=min(8, int(os.cpu_count() * 2 / 3)),  # Use 8 workers minimum or 2/3 of available CPU cores

        # Enable full evaluation metrics logging
        prediction_loss_only=False,  # Compute all metrics during evaluation

        # find a batch size that will fit into memory automatically through exponential decay,
        # avoiding CUDA Out-of-Memory errors
        # auto_find_batch_size=True,

        # Optimizer settings for memory efficiency
        # Using StableAdamW because it was used by recent BERT-based models like ModernBERT
        # https://github.com/huggingface/transformers/blob/main/docs/source/en/optimizers.md
        optim="stable_adamw",

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_dataset["train"],
        eval_dataset=train_test_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    latest_checkpoint = find_latest_checkpoint(output_dir)

    if latest_checkpoint:
        logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        trainer.train()


def get_max_memory(reserve_gb=2.0, fraction=0.85):
    """
    Auto-generate max_memory dict for all available GPUs.

    Args:
        reserve_gb: GB to reserve for PyTorch overhead
        fraction: Fraction of remaining memory to use (0-1)

    Returns:
        max_memory dict for model loading
    """
    if not torch.cuda.is_available():
        return {}

    max_memory = {}

    for i in range(torch.cuda.device_count()):
        # Get total GPU memory in GB
        total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)

        # Calculate usable memory
        usable_gb = (total_gb - reserve_gb) * fraction
        max_memory[i] = f"{usable_gb:.1f}GB"

        logger.info(f"GPU {i}: {total_gb:.1f}GB total, allocating {usable_gb:.1f}GB")

    return max_memory


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_names", type=str, nargs='+',
                        default=["ExponentialScience/dlt-articles-ner-filtered"],
                        help="List of dataset names to load and combine. "
                             "Can specify multiple datasets separated by spaces.")


    parser.add_argument("--model_id", type=str,
                        default="allenai/scibert_scivocab_cased")

    # Using 512 sequence length based on bert: https://arxiv.org/pdf/2412.13663
    parser.add_argument("--block_size", type=int,
                        default=512)

    parser.add_argument("--models_dir", type=str,
                        default="./../../models")

    parser.add_argument("--hf_datasets", type=str,
                        default="./../../data/hf_datasets")

    # Latest checkpoint in case of resuming directly
    parser.add_argument("--latest_checkpoint", type=str, default=None,
                        help="Path to the latest checkpoint to resume training from. "
                             "If not provided, training will start from scratch.")

    parser.add_argument("--num_train_epochs", type=int, default=3)

    parser.add_argument("--set_per_process_memory_fraction", type=float,
                        default=0.95,
                        help="GPU memory fraction to use per process")

    parser.add_argument("--seed", type=int,
                        default=42,
                        help="Random seed for reproducibility")

    parser.add_argument("--run_name", type=str,
                        default=None,
                        help="Run name for logging (e.g., WandB). If not provided, a timestamp will be used.")

    args = parser.parse_args()

    # Set seed for reproducibility
    logger.info(f"Setting seed to {args.seed} for reproducibility")
    set_seed(args.seed)  # This sets seed for transformers, numpy, torch, and random

    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # # Check if models_dir exist
    # if not os.path.isdir(args.models_dir):
    #     logger.info("Models directory does not exist")
    #     exit(1)
    os.makedirs(args.models_dir, exist_ok=True)

    # Make sure that it has been logged in to Hugging Face
    logger.info("Checking Hugging Face login status...")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        logger.info("Logged in to Hugging Face")
    else:
        logger.warning("No HF_TOKEN found. Some models may not be accessible.")

    # logger.info(f"Loading the dataset: {args.dataset_name}")
    # # ds = load_dataset(args.dataset_name, split="train")
    # ds = load_dataset_safe(args.hf_datasets, args.dataset_name)

    logger.info(f"Loading and combining datasets: {args.dataset_names}")
    # Load and combine multiple datasets
    ds = load_and_combine_datasets(args.hf_datasets, args.dataset_names)

    num_cuda_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_cuda_gpus} CUDA GPUs")

    logger.info(f"Loading the model: {args.model_id}")

    if args.latest_checkpoint is not None:

        logger.info(f"Loading model and tokenizer from checkpoint: {args.latest_checkpoint}")
        model = AutoModelForMaskedLM.from_pretrained(
            args.latest_checkpoint,
            torch_dtype=torch.bfloat16,
            # IMPORTANT: DO NOT USE device_map when using distributed training
            # device_map="balanced",  # Automatically distribute across available GPUs. Same as "auto"
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            # max_memory=get_max_memory(reserve_gb=2, fraction=0.85),  # Leave headroom of 2 GBs per GPU for PyTorch overhead
        )
        tokenizer = AutoTokenizer.from_pretrained(args.latest_checkpoint)

    else:

        logger.info(f"Loading model and tokenizer from Hugging Face: {args.model_id}")
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            # IMPORTANT: DO NOT USE device_map when using distributed training
            # device_map="balanced",  # Automatically distribute across available GPUs. Same as "auto"
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            # max_memory=get_max_memory(reserve_gb=2, fraction=0.85),  # Leave headroom of 2 GBs per GPU for PyTorch overhead
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    lm_dataset = gen_lm_dataset(ds, tokenizer, args.block_size)

    logger.info(f"Dataset size after grouping: {len(lm_dataset)}")
    tokenizer.pad_token = tokenizer.sep_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    # mlm_probability=0.3,
                                                    mlm_probability=0.15, # Using 15% masking based on BERT training
                                                    )

    logger.info(f"Train-test splitting the dataset with 10% for testing")
    train_test_dataset = lm_dataset.train_test_split(test_size=0.1,
                                                    seed=args.seed)

    # Print memory usage before training
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f}GB allocated, "
                      f"{torch.cuda.memory_reserved(i) / 1024 ** 3:.2f}GB reserved")

    logger.info(f"Training the model on the dataset...")
    date_time_run = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.models_dir, "LedgerBERT", date_time_run)
    os.makedirs(output_dir, exist_ok=True)

    train(model, tokenizer, data_collator, train_test_dataset, output_dir, args,
          set_per_process_memory_fraction=args.set_per_process_memory_fraction)

    # Solves warning: "destroy_process_group() was not called before program exit"
    # in case of distributed training with multiple GPUs using accelerate
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()