import argparse
import logging
from models.tasks.dlt_classifier import load_dataset_safe
from datasets import Dataset, Features, Value
from sklearn.metrics import confusion_matrix



# Configure logging
logger = logging.getLogger(__name__)

def eval_performance(df):

    # Get confusion matrix components
    tn, fp, fn, tp = confusion_matrix(df['class'], df['median_pred']).ravel()
    # Print the metrics
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_datasets", type=str,
                        default="./../../data/hf_datasets")

    parser.add_argument("--gold_column", type=str, default="class",
                        help="Column name containing gold standard labels")

    parser.add_argument("--dataset_name", type=str,
                        default="ExponentialScience/DLT-Open-Access-Literature-annotated-entities",
                        choices=[
                            # Represents the 100 hand label used to test the performance using max prediction score
                            "ExponentialScience/dlt-articles-classification",
                            # Dataset filtered for English content and content with more than 500 tokens
                            # and less than 40k tokens to avoid very short and very long texts
                            "ExponentialScience/DLT-Open-Access-Literature-annotated-entities" #
                        ]
                        )

    parser.add_argument("--hf_dataset_repo", type=str,
                        default="ExponentialScience/dlt-articles-ner-filtered")

    args = parser.parse_args()

    ds = load_dataset_safe(args.hf_datasets, args.dataset_name)

    logger.info(f"Number of samples: {len(ds)}")

    # Capture the original features before converting to pandas
    original_features = ds.features
    logger.info(f"Original features: {original_features}")

    # Get column names to preserve
    original_columns = list(ds.column_names)

    df = ds.to_pandas()

    logger.info("Converted dataset to pandas DataFrame")

    # Count the number of predictions in each row
    df["num_preds"] = df["predictions"].str.len()

    logger.info("Filtering dataset...")
    # Get the prediction scores
    df["prediction_scores"] = df["predictions"].apply(
        lambda x: [y["score"] for y in x if y["entity_group"] != "ESG"]
    )

    # Get the median and max prediction score for a row
    df["prediction_max"] = df["prediction_scores"].apply(lambda x: max(x) if x else None)
    # df["prediction_median"] = df["prediction_scores"].apply(lambda x: pd.Series(x).median() if x else None)

    # Get the number of max predictions with a score above 0.995
    df["max_pred"] = df["prediction_max"] > 0.995

    # To validate the scoring performance versus the hand labels, we can use the class column
    if args.gold_column in df.columns:
        logger.info(f"Evaluating performance using gold column '{args.gold_column}'")
        eval_performance(df)

    # Remove the data that is False in max_pred
    df_filtered = df[df["max_pred"] == True].copy()

    logger.info("Converting DataFrame back to Hugging Face dataset...")
    # Create features dictionary combining original features with new columns
    features_dict = {}

    # Add original features
    for col_name in original_columns:
        if col_name in original_features:
            features_dict[col_name] = original_features[col_name]

    # Add new features
    features_dict.update({
        'num_preds': Value('int32'),
        'prediction_scores': [Value('float32')],  # List of floats
        'prediction_max': Value('float32'),
        # 'prediction_median': Value('float32'),
        'max_pred': Value('bool')
    })

    # Create Features object
    features = Features(features_dict)

    # Convert the DataFrame back to a Hugging Face dataset
    hf_dataset = Dataset.from_pandas(df_filtered, features=features, preserve_index=False)

    logger.info(f"Number of samples after filtering: {len(hf_dataset)}")
    logger.info(f"Final features: {hf_dataset.features}")

    # Save the filtered dataset to the specified Hugging Face repository
    hf_dataset.push_to_hub(args.hf_dataset_repo, private=True)

    logger.info(f"Dataset pushed to {args.hf_dataset_repo}")


if __name__ == "__main__":
    main()