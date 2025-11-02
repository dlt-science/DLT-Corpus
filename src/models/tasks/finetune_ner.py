import os
import evaluate
import numpy as np
import pandas as pd
import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import Accelerator
import logging
import wandb
from functools import partial
from multiprocessing import cpu_count
from collections import Counter

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from datasets import Dataset, Features, ClassLabel, Sequence, Value
from transformers import Trainer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments, AutoConfig, set_seed, logging as transformers_logging, EarlyStoppingCallback

# Adding the path to be able to import the analytics module
import sys

sys.path.append('./../../')
from models.continual_training import find_latest_checkpoint, load_dataset_safe

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Enable transformers logging
transformers_logging.enable_progress_bar()
transformers_logging.set_verbosity_info()

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Optimize for multi-core CPU usage
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

os.environ["WANDB_PROJECT"] = os.environ["WANDB_PROJECT_NER"]  # name your W&B project

# # log all model checkpoints by saving them to W&B as artifacts
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"

torch.set_num_threads(4)

# Check CUDA availability
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} CUDA GPUs")
    if num_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(num_gpus))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Function to compute metrics for NER task
label_list = []  # This should be properly initialized


def compute_metrics(p):
    """
    Metrics using the HuggingFace datasets library (https://huggingface.co/docs/evaluate/index)
    that uses seqeval under the roof: https://huggingface.co/spaces/evaluate-metric/seqeval

    """
    metric = evaluate.load("seqeval")

    predictions, labels = p

    # Convert logits to index of max logit
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results_phrase = metric.compute(
        predictions=true_predictions, references=true_labels, mode=None)

    results_token = metric.compute(
        predictions=true_predictions, references=true_labels,  mode="strict")

    return {
        "precision_token": results_token["overall_precision"],
        "recall_token": results_token["overall_recall"],
        "f1_token": results_token["overall_f1"],
        "accuracy_token": results_token["overall_accuracy"],
        "precision_phrase": results_phrase["overall_precision"],
        "recall_phrase": results_phrase["overall_recall"],
        "f1_phrase": results_phrase["overall_f1"],
        "accuracy_phrase": results_phrase["overall_accuracy"],
    }


def tokenize_and_align_labels(examples, tokenizer, args):
    """
    Tokenizes the input examples and aligns the labels with the tokenized inputs.
    :param examples:
    :return:
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,  # Add padding for consistent batch sizes
        max_length=args.max_seq_length,  # Use the configured max length
        is_split_into_words=True # Handles tokenized inputs correctly
    )

    labels = []

    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def get_stratification_labels(dataset):
    """
    Create stratification labels for NER dataset based on entity type distribution.
    For NER, we'll use the most frequent entity type in each document as the stratification label.
    """
    stratification_labels = []

    for _, row in dataset.iterrows():
        # Get the NER tags for this sample, excluding special tokens (-100)
        ner_tags = [tag for tag in row['labels'] if tag != -100]

        if not ner_tags:
            # If no valid tags, assign to a default class (O tag usually has id 0)
            stratification_labels.append(0)
        else:
            # Count the occurrences of each entity type
            tag_counts = Counter(ner_tags)
            # Use the most frequent entity type as stratification label
            most_frequent_tag = tag_counts.most_common(1)[0][0]
            stratification_labels.append(most_frequent_tag)

    return np.array(stratification_labels)

if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="./../../../models",
                        help="Directory to save models")

    parser.add_argument("--data_task_dir", type=str,
                        default="./../../../data/tasks/ner",
                        help="Directory containing NER task data")

    parser.add_argument("--hf_datasets", type=str,
                        default="./../../../data/hf_datasets")

    parser.add_argument('--dataset_name', type=str,
                        default="ExponentialScience/ESG-DLT-NER",
                        help='Path to the dataset')

    parser.add_argument("--model_name", type=str,
                        default="allenai/scibert_scivocab_cased",
                        help="Model name or path to load from")

    parser.add_argument("--tokenizer_name", type=str,
                        default=None,
                        help="Model name or path to load from")

    parser.add_argument("--num_epochs", type=int,
                        default=20,
                        help="Number of training epochs per fold")

    parser.add_argument("--learning_rate", type=float,
                        # default=5e-5,
                        default=1e-5,  # To reduce the evaluation loss
                        help="Learning rate")

    parser.add_argument("--train_batch_size", type=int,
                        default=16,
                        help="Training batch size per device")

    parser.add_argument("--eval_batch_size", type=int,
                        default=32,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=2,
                        help="Gradient accumulation steps")

    parser.add_argument("--n_splits", type=int,
                        default=5,
                        help="Number of K-fold splits")

    parser.add_argument("--warmup_steps", type=int,
                        default=500, # 10% of the entire training steps
                        help="Number of warmup steps")

    parser.add_argument("--max_seq_length", type=int,
                        default=512,
                        help="Maximum sequence length")

    # Add checkpoint path argument for loading from local path
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to a local checkpoint to load model and tokenizer from")

    parser.add_argument("--set_per_process_memory_fraction", type=float,
                        default=0.90,
                        help="GPU memory fraction to use per process")

    parser.add_argument("--tokenized_dataset_name", type=str,
                        default="dataset_ModernBERT-base.pkl",
                        help="GPU memory fraction to use per process")

    parser.add_argument("--early_stopping_patience", type=int,
                        default=5,
                        help="Number of epochs with no improvement after which training will be stopped")

    parser.add_argument("--early_stopping_threshold", type=float,
                        default=0.0001,
                        help="Minimum change in the monitored metric to qualify as an improvement")

    parser.add_argument("--metric_for_best_model", type=str,
                        default="eval_f1_phrase",
                        help="Metric to use for early stopping and best model selection")

    parser.add_argument("--seed", type=int,
                        default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set seed for reproducibility
    logger.info(f"Setting seed to {args.seed} for reproducibility")
    # This sets seed for transformers, numpy, torch, and random
    set_seed(args.seed)

    logger.info(f"Is CUDA available: {torch.cuda.is_available()}")

    # Clean up the GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"Type of device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")

    # Initialize accelerator
    accelerator = Accelerator()
    num_gpus = accelerator.num_processes
    logger.info(f"Using {num_gpus} GPUs for training")

    # Set memory fraction if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(
            f"Setting GPU memory fraction to {args.set_per_process_memory_fraction}")
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(
                args.set_per_process_memory_fraction, device=i)

    if args.checkpoint_path:
        logger.info(
            f"Loading the tokenizer from checkpoint {args.checkpoint_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint_path, use_fast=True)

    else:
        tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name
        logger.info(f"Loading the tokenizer from {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True)

    # set model_max_length to 512 as label texts are no longer than 512 tokens
    tokenizer.model_max_length = args.max_seq_length

    # Load the dataset from the HuggingFace datasets library
    dataset = None

    ds = load_dataset_safe(
        args.hf_datasets, args.dataset_name
    )

    logger.info("Aligning the labels with the tokenized inputs...")

    # Pass multiple arguments to the preprocess function
    tokenize_and_align_labels_fn = partial(tokenize_and_align_labels,
                                           tokenizer=tokenizer, args=args)

    dataset = ds.map(tokenize_and_align_labels_fn,
                     batched=True,
                     num_proc=int(cpu_count() * 2/3))

    logger.info("Converting the dataset to pandas DataFrame...")
    # Convert the dataset to a pandas DataFrame
    dataset = dataset.to_pandas()

    # if args.model_name == 'bert-base-cased':
    #     logger.info(f"Loading the dataset from HuggingFace...")
    #     dataset = pd.read_parquet("hf://datasets/ExponentialScience/ESG-DLT-NER/data/train-00000-of-00001.parquet")
    #
    # if dataset is None:
    #     if not args.dataset_name:
    #         args.dataset_name = f"dataset_{args.model_name.split('/')[-1]}.pkl"
    #
    #     dataset_path = os.path.join(args.data_task_dir, args.dataset_name)
    #     logger.info(f"Loading the dataset from {dataset_path}...")
    #     dataset = pd.read_pickle(dataset_path)

    # # Convert format of certain columns to list
    # dataset["input_ids"] = dataset["input_ids"].apply(lambda x: list(x))
    # dataset["attention_mask"] = dataset["attention_mask"].apply(lambda x: list(x))
    # dataset["labels"] = dataset["labels"].apply(lambda x: list(x))
    # dataset["ner_tags"] = dataset["ner_tags"].apply(lambda x: list(x))
    # dataset["tokens"] = dataset["tokens"].apply(lambda x: list(x))

    # Drop columns not needed
    cols_use = [c for c in dataset.columns if c not in [
        "__index_level_0__", "text", "ner_tags", "tokens"]]
    dataset = dataset[cols_use]
    # dataset.drop(columns=["__index_level_0__", "text"], inplace=True)

    logger.info(
        f"Loading the label_to_id and id_to_label jsons from {args.data_task_dir}...")
    with open(os.path.join(args.data_task_dir, "label_to_id.json"), "r") as f:
        label_to_id = json.load(f)

    id_to_label = {v: k for k, v in label_to_id.items()}

    # Get the unique labels
    unique_labels = list(set(label_to_id.keys()))

    # Get the label_list
    label_list = list(label_to_id.keys())

    # Set the hyperparameters
    # num_epochs = 20  # for each k-fold
    # learning_rate = 5e-5
    # train_batch_size = 32
    # eval_batch_size = 64

    # train_batch_size = 16  # Reduced from 32
    # eval_batch_size = 32  # Reduced from 64

    # early_stopping_patience = 3

    # Add the time to the output directory
    date_time_run = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.model_dir, args.model_name.split("/")[-1], date_time_run)

    # use some warmup steps to increase the learning rate up to a certain point
    # and then use your normal learning rate afterwards
    # warmup_steps = 500
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    logger.info(f"Defining the training args...")
    # define training args
    # Based on: https://huggingface.co/docs/transformers/v4.53.2/en/main_classes/trainer
    training_args = TrainingArguments(

        output_dir=output_dir,
        logging_dir=logs_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,

        do_train=True,
        do_eval=True,

        # Evaluation settings
        eval_strategy="epoch",
        eval_accumulation_steps=2,
        eval_steps=100,

        # # Regularization
        # #  Label smoothin introduces noise for the labels.
        # #  This accounts for the fact that datasets may have mistakes in them
        # # https://paperswithcode.com/method/label-smoothing
        # label_smoothing_factor=0.1,
        # weight_decay=0.01,

        # # Early stopping only (no best model loading)
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,  # F1 score is better when higher
        # save_safetensors=True,  # Use safetensors format for better compatibility

        # Checkpoint management
        save_strategy="epoch",
        save_steps=100,  # Save every 100 steps
        save_total_limit=5,  # Reduced to save space, early stopping will handle best model
        save_on_each_node=False,  # Only save on main process to avoid conflicts

        # Memory management
        remove_unused_columns=False,
        dataloader_pin_memory=True,  # Enable pin memory with sufficient RAM
        # Use 8 workers minimum or half of available CPU cores
        dataloader_num_workers=min(8, cpu_count() // 2),

        # Logging
        logging_steps=10,
        logging_strategy="steps",
        # report_to="tensorboard",

        run_name=args.model_name.split('/')[-1],
        report_to="wandb",  # enable logging to W&B

        # allows to effectively have a larger batch size while using less memory per step
        # # Use bfloat16 for better memory efficiency
        # fp16=False if torch.backends.mps.is_available() else True,  # Enable mixed precision training
        # Enable bf16 if supported
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        bf16_full_eval=True,

        # stable_adamw is the StableAdamW optimizer from ModernBERT:
        # https://github.com/huggingface/transformers/pull/36606/commits/7b0e57c6c220045d51de7a450d06501f807748ab
        # https://github.com/AnswerDotAI/ModernBERT/blob/main/src/optimizer.py
        optim="adamw_torch_fused",
        # local_rank=0,  # Enable distributed training. It is
        use_mps_device=True if torch.backends.mps.is_available() else False,

        # DDP settings - Fix for unused parameters warning
        ddp_find_unused_parameters=False,
    )

    logger.info(f"Setting up the config for the model...")
    # Set up config for the model
    # Using reference_compile=False to avoid issues with torch.compile
    # and to ensure compatibility with the current version of transformers
    # https://huggingface.co/answerdotai/ModernBERT-base/discussions/14
    config = AutoConfig.from_pretrained(
        args.checkpoint_path if args.checkpoint_path else args.model_name,
        num_labels=len(unique_labels),
        label2id=label_to_id,
        id2label=id_to_label,

        # See why use bfloat16: https://www.cerebras.ai/blog/to-bfloat-or-not-to-bfloat-that-is-the-question
        # to use less memory and speed up training
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        # device_map="auto", # Remove to be able to use accelerate
        reference_compile=False,  # Avoid issues with torch.compile
        # attn_implementation="flash_attention_2"
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Clear the GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize the GroupKFold class
    # n_splits = 5
    group_kfold = GroupKFold(n_splits=args.n_splits)
    # stratified_group_kfold = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    #
    # # Initialize the StratifiedGroupKFold class
    # logger.info(f"Creating stratification labels for balanced class distribution...")
    # stratification_labels = get_stratification_labels(dataset)
    #
    # logger.info(f"Class distribution in stratification labels:")
    # unique_labels_count = Counter(stratification_labels)
    # for label, count in sorted(unique_labels_count.items()):
    #     logger.info(f"  Class {label}: {count} samples ({count / len(stratification_labels) * 100:.1f}%)")

    logger.info(f"Defining the features...")

    # Define features
    features = Features({
        'input_ids': Sequence(Value('int64')),
        'attention_mask': Sequence(feature=Value(dtype='int64')),
        'labels': Sequence(ClassLabel(num_classes=len(unique_labels), names=unique_labels)),
        # 'ner_tags': Sequence(ClassLabel(num_classes=len(unique_labels), names=unique_labels)),
        # 'tokens': Sequence(feature=Value(dtype='string')),
        # 'paper_name': Value(dtype='string')
    })

    # Store results from each fold
    results = []
    evaluations = []

    # Clear the GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # Store results from each fold
    results = []
    evaluations = []
    scores = []

    logger.info(f"Starting the training loop...")
    fold_number = 1
    dfs_tokens = []
    dfs_phrases = []

    # # Check for existing checkpoint
    # latest_checkpoint = find_latest_checkpoint(training_args.output_dir)

    cols_training = ['input_ids', 'attention_mask', 'labels']
    for train_index, val_index in tqdm(group_kfold.split(dataset, groups=dataset['paper_name']), total=args.n_splits):
    # for train_index, val_index in tqdm(
    #         stratified_group_kfold.split(dataset, stratification_labels, groups=dataset['paper_name']),
    #         total=args.n_splits
    # ):

        logger.info(f"Preparing for fold {fold_number}...")

        # # Log class distribution for this fold
        # train_labels = stratification_labels[train_index]
        # val_labels = stratification_labels[val_index]
        #
        # train_distribution = Counter(train_labels)
        # val_distribution = Counter(val_labels)
        #
        # logger.info(f"Fold {fold_number} - Training set class distribution:")
        # for label, count in sorted(train_distribution.items()):
        #     logger.info(f"  Class {label}: {count} samples ({count / len(train_labels) * 100:.1f}%)")
        #
        # logger.info(f"Fold {fold_number} - Validation set class distribution:")
        # for label, count in sorted(val_distribution.items()):
        #     logger.info(f"  Class {label}: {count} samples ({count / len(val_labels) * 100:.1f}%)")

        # Load the model
        logger.info(f"Loading the model {args.model_name}...")

        # # Using to 'cuda' to ensure the model is loaded on the GPU and be able to use flash attention
        # model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config).to('cuda')
        # Reinitialize model for each fold
        if args.checkpoint_path:
            logger.info(
                f"Loading the model from checkpoint {args.checkpoint_path}...")
            model = AutoModelForTokenClassification.from_pretrained(
                args.checkpoint_path, config=config).to('cuda')
        else:
            logger.info(f"Loading the model from {args.model_name}...")
            model = AutoModelForTokenClassification.from_pretrained(
                args.model_name, config=config).to('cuda')

        # Count and log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)

        logger.info(f"Model loaded successfully!")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        logger.info(
            f"Model size: ~{total_params * 4 / (1024 ** 3):.2f} GB (assuming float32)")

        # Create and update fold-specific output and logging directories
        training_args.output_dir = os.path.join(output_dir, f"fold_{fold_number}")
        os.makedirs(training_args.output_dir, exist_ok=True)
        training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
        os.makedirs(training_args.logging_dir, exist_ok=True)

        # Update the fold name in the training arguments for W&B logging
        training_args.run_name = f"{args.model_name}-fold-{fold_number}"

        # Split data into training and validation
        train_data, val_data = dataset.iloc[train_index], dataset.iloc[val_index]

        # Create training and validation datasets
        train_dataset = Dataset.from_pandas(
            train_data[cols_training], features=features, preserve_index=False)
        eval_dataset = Dataset.from_pandas(
            val_data[cols_training], features=features, preserve_index=False)

        # # Create early stopping callback
        # early_stopping_callback = EarlyStoppingCallback(
        #     early_stopping_patience=args.early_stopping_patience,
        #     early_stopping_threshold=args.early_stopping_threshold
        # )

        logger.info(f"Creating Trainer instance for fold {fold_number}...")

        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # callbacks=[early_stopping_callback],
        )

        # if latest_checkpoint:
        #     logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        #     trainer.train(resume_from_checkpoint=latest_checkpoint)
        # else:
        # trainer.train()

        logger.info(f"Starting training for fold {fold_number}...")
        trainer.train()

        # Make predictions
        predictions, labels, metrics = trainer.predict(eval_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Save the scores for the models:
        scores.append(metrics)

        # Separate overall and detailed metrics
        token_metrics = {k: v for k,
                         v in metrics.items() if k.endswith("_token")}
        phrase_metrics = {k: v for k,
                          v in metrics.items() if k.endswith("_phrase")}

        df_token = pd.DataFrame([token_metrics])
        df_phrase = pd.DataFrame([phrase_metrics])

        # Add the fold number to the DataFrames
        df_token['fold'] = fold_number
        df_phrase['fold'] = fold_number

        # Append the DataFrames to the list
        dfs_tokens.append(df_token)
        dfs_phrases.append(df_phrase)

        # Close previous W&B run if it exists
        if wandb.run is not None:
            logger.info(f"Closing previous W&B run for fold {fold_number}...")
            wandb.finish()

        fold_number += 1

    # Concatenate the DataFrames
    df_overall_token = pd.concat(dfs_tokens)
    df_overall_phrase = pd.concat(dfs_phrases)

    # Save the DataFrames to CSV files
    logger.info(f"Saving the metrics to CSV files...")
    df_overall_token.to_csv(os.path.join(output_dir, f"{args.model_name.split('/')[-1]}_overall-token.csv"),
                            index=False)
    df_overall_phrase.to_csv(os.path.join(output_dir, f"{args.model_name.split('/')[-1]}_overall-phrase.csv"),
                             index=False)

    # Save the model to local storage
    logger.info(f"Saving the model to {output_dir}...")
    model.save_pretrained(output_dir)

    # Save the tokenizer to local storage
    logger.info(f"Saving the tokenizer to {output_dir}...")
    tokenizer.save_pretrained(output_dir)

    # Solves warning: "destroy_process_group() was not called before program exit"
    # in case of distributed training with multiple GPUs using accelerate
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
