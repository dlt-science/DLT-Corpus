from transformers import AutoModel, AutoConfig
from functools import reduce
import argparse
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_model(model_name: str) -> AutoModel:
    """Load a Hugging Face model by name."""
    try:
        logger.info(f"Loading model: {model_name}")
        return AutoModel.from_pretrained(model_name)
    except Exception as e:
        logger.info(f"Error loading model {model_name}: {e}")
        sys.exit(1)


def get_context_window_size(model_name: str) -> int:
    """Get the context window size from model configuration."""
    try:
        config = AutoConfig.from_pretrained(model_name)
        # Common attribute names for context window size
        context_attrs = ['max_position_embeddings', 'n_positions', 'max_seq_len', 'seq_length']

        for attr in context_attrs:
            if hasattr(config, attr):
                return getattr(config, attr)

        logger.warning("Context window size not found in model configuration")
        return None
    except Exception as e:
        logger.error(f"Error getting context window size: {e}")
        return None


def format_number(num: int) -> str:

    if num >= 1_000_000_000:
        return f"{round(num / 1_000_000_000)}B"
    elif num >= 1_000_000:
        return f"{round(num / 1_000_000)}M"
    elif num >= 1_000:
        return f"{round(num / 1_000)}K"
    else:
        return str(num)


def main() -> None:
    """Main function to calculate and display model parameters."""
    parser = argparse.ArgumentParser(description="Calculate total parameters of a Hugging Face model")
    parser.add_argument("--model_name", help="Name of the Hugging Face model")
    args = parser.parse_args()

    model = load_model(args.model_name)

    logger.info(f"Calculating total parameters for {args.model_name}")
    # Calculate total parameters using functional programming
    total_params = reduce(lambda total, param: total + param.numel(), model.parameters(), 0)

    # Get context window size
    context_size = get_context_window_size(args.model_name)

    # Display results
    formatted_count = format_number(total_params)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Total Parameters: {total_params:,} ({formatted_count})")

    if context_size:
        logger.info(f"Context Window Size: {context_size:,} tokens")
    else:
        logger.info("Context Window Size: Not available")


if __name__ == "__main__":
    main()