import pandas as pd
from datasets import Dataset, Value, ClassLabel, Features

import logging
import argparse
import os

# Configure logging
logger = logging.getLogger(__name__)

def get_sentiment_labels(df, sentiment_col, label_cols, percentiles):
    """
    Sample by the top 75% percentile for the positive news and 25% percentile on the same column of positive
    news to get the negative labels. Then, the range between 25% and 75% represents neutral zone.
    The use of percentiles allows to avoid the bias of the total votes, which can be skewed by popularity
    :param df:
    :param label_cols:
    :param percentiles:
    :return:
    """

    # Calculate the total votes for the columns
    df['total_votes'] = df[label_cols].sum(axis=1)

    # Normalize the vote counts to get the percentage
    for col in label_cols:
        df[f"{col}_pct"] = df[col] / df['total_votes'] * 100

    # Single column to find the 25 and 75 percentiles
    col_filter = percentiles[75]

    p25 = df[f"{col_filter}_pct"].quantile(0.25)
    p75 = df[f"{col_filter}_pct"].quantile(0.75)

    # # Rank by sentiment (this normalizes by distribution, not vote count)
    # df['sentiment_rank'] = df[f"{col_filter}_pct"].rank(pct=True) * 100

    # First set all classifications to neutral
    df[sentiment_col] = "neutral"

    # Then set the classifications based on the percentiles of the main column used as based for classification
    df.loc[df[f"{col_filter}_pct"] <= p25, sentiment_col] = percentiles[25]
    df.loc[df[f"{col_filter}_pct"] >= p75, sentiment_col] = percentiles[75]

    return df


def preprocess_dataset(df):
    """
    Apply quality control filters and basic preprocessing.

    Args:
        df: Raw input DataFrame
        min_votes: Minimum total votes required for inclusion

    Returns:
        Filtered and preprocessed DataFrame
    """
    # Calculate total votes per article
    vote_columns = ['bearish', 'bullish', 'liked', 'disliked', 'important', 'lol']

    df['total_votes'] = df[vote_columns].sum(axis=1)

    # Filter for votes more than 0
    df = df[df['total_votes'] > 0].copy()

    # Get the median of the total votes for each category
    market = ['bearish', 'bullish']
    engagement = ['liked', 'disliked']
    content = ['important', 'lol']

    df['votes_mkt'] = df[market].sum(axis=1)
    df['votes_engagement'] = df[engagement].sum(axis=1)
    df['votes_content'] = df[content].sum(axis=1)

    min_median = min(df['votes_mkt'].median(), df['votes_engagement'].median(), df['votes_content'].median())

    # Filter the total votes based on the median
    df_filtered = df[(df['total_votes'] >= min_median)].copy()

    # Apply quality filters
    # df_filtered = df[df['total_votes'] >= min_votes].copy()
    df_filtered = df_filtered.dropna(subset=['title']).copy()
    df_filtered['description'] = df_filtered['description'].fillna('')
    
    # Drop the rows with 5 total votes
    df_filtered = df_filtered[df_filtered['total_votes'] > 0].copy()

    # Remove news with no description
    df_filtered = df_filtered[df_filtered['description'].str.strip() != ''].copy()

    # Remove nes with NaN or Null in description
    df_filtered = df_filtered[(df_filtered['description'].notna()) & (df_filtered['description'] != 'null')].copy()

    # Remove rows with empty description
    df_filtered = df_filtered[df_filtered['description'].str.len() > 0]

    # Remove rows containing "RT @" at the beginning of the description
    df_filtered = df_filtered[~df_filtered['description'].str.startswith('RT @')].copy()

    # Remove rows only containing "-" in the description
    df_filtered = df_filtered[df_filtered["description"] != "-"]

    logger.info(f"Dataset filtered: {len(df)} → {len(df_filtered)} articles")

    return df_filtered


def transform_to_hf_format(df):
    """
    Transform DataFrame to HuggingFace-compatible format with hierarchical labels.

    Args:
        df: Preprocessed input DataFrame

    Returns:
        List of dictionaries in HuggingFace format
    """
    transformed_data = []

    market_direction = {25: 'bearish', 75: 'bullish'}
    engagement_quality = {25: 'disliked', 75: 'liked'}
    content_characteristics = {25: 'lol', 75: 'important'}

    for sentiment_col, cols in [
        ('market_direction', market_direction),
        ('engagement_quality', engagement_quality),
        ('content_characteristics', content_characteristics)
    ]:
        logger.info(f"Labelling sentiment category: {sentiment_col}")
        df = get_sentiment_labels(df, sentiment_col=sentiment_col,
                                  label_cols=list(cols.values()),
                                  percentiles=cols)

    # Convert to list of dictionaries using to_dict('records')
    records = df.to_dict('records')

    # Transform to final format
    transformed_data = [
        {
            # 'id': str(row['id']),
            'timestamp': str(row['newsDatetime']) if pd.notna(row['newsDatetime']) else '',
            'title': str(row['title']),
            'description': str(row['description']),
            'text': str(f"{row['title'].strip()}\n\n{row['description'].strip()}".strip()),
            'market_direction': row['market_direction'],
            'engagement_quality': row['engagement_quality'],
            'content_characteristics': row['content_characteristics'],
            'vote_counts': {
                'bearish': int(row['bearish']),
                'bullish': int(row['bullish']),
                'liked': int(row['liked']),
                'disliked': int(row['disliked']),
                'important': int(row['important']),
                'lol': int(row['lol']),
                # 'toxic': int(row['toxic'])
            },
            'total_votes': int(row['total_votes']),
            'source_url': str(row['sourceUrl']) if pd.notna(row['sourceUrl']) else '',
            'url': str(row['url']) if pd.notna(row['url']) else '',
        }
        for row in records
    ]

    return transformed_data


def create_hf_dataset(df):
    """
    Main function to create HuggingFace dataset from cryptocurrency news CSV.

    Args:
        csv_path: Path to input CSV file
        output_path: Local output path (optional)
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_name: Repository name for Hub upload

    Returns:
        HuggingFace DatasetDict with train/validation/test splits
    """
    # Define HuggingFace features schema

    # The toxic column is not used because there are too few samples (around 200) and it is not representative
    # of the sentiment classification task. It is better to focus on the main sentiment labels.
    features = Features({
        # 'id': Value('string'),
        'timestamp': Value('string'),
        'title': Value('string'),
        'description': Value('string'),
        'text': Value('string'),
        'market_direction': ClassLabel(names=['neutral', 'bearish', 'bullish']),
        'engagement_quality': ClassLabel(names=['neutral', 'liked', 'disliked']),
        'content_characteristics': ClassLabel(names=['neutral', 'important', 'lol']),
        'vote_counts': {
            'bearish': Value('int32'),
            'bullish': Value('int32'),
            'liked': Value('int32'),
            'disliked': Value('int32'),
            'important': Value('int32'),
            'lol': Value('int32'),
            # 'toxic': Value('int32')
        },
        'total_votes': Value('int32'),
        'source_url': Value('string'),
        'url': Value('string'),
    })

    # Load and preprocess data
    filtered_df = preprocess_dataset(df)

    # Transform to HuggingFace format
    transformed_data = transform_to_hf_format(filtered_df)
    transformed_df = pd.DataFrame(transformed_data)

    dataset = Dataset.from_pandas(transformed_df, features=features)


    return dataset

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--task_path', type=str,
                        default="./../../data/tasks/sentiment",
                        help='Path to save the output Markdown files.')

    parser.add_argument('--hf_datasets', type=str,
                        default="./../../data/hf_datasets",
                        help='Path to save the output Markdown files.')

    parser.add_argument("--hf_dataset_repo", type=str,
                        default="ExponentialScience/DLT-Sentiment-News")

    args = parser.parse_args()

    sentiment_csv_path = os.path.join(args.task_path, "cryptopanic_news.csv")

    df = pd.read_csv(sentiment_csv_path)

    # rename some of the columns
    df.rename(columns={
        'negative': 'bearish',
        'positive': 'bullish'
    }, inplace=True)

    # Dataset Features Schema
    hf_dataset = create_hf_dataset(df)

    # Save dataset locally
    hf_dataset_path = os.path.join(args.hf_datasets, "DLT-Sentiment-News")

    logger.info(f"Writing HF dataset locally to {hf_dataset_path}")
    hf_dataset.save_to_disk(hf_dataset_path)

    logger.info(f"Pushing dataset to {args.hf_dataset_repo}...")
    hf_dataset.push_to_hub(args.hf_dataset_repo, token=True, private=True)



if __name__ == "__main__":
    main()