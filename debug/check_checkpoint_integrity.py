import torch
import os
import sys
import argparse
import logging

# Adjust the following import based on your project structure
# Ensure that your project directory is in the Python path
# For example, if your project structure is:
# my_project/
# ├── models/
# │   └── my_model.py
# └── scripts/
#     └── check_and_fix_checkpoint.py
#
# You might need to append the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_project.models import MyModel  # Replace with your actual model class

def setup_logging():
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_and_fix_checkpoint(original_checkpoint_path, fixed_checkpoint_path, device='cpu'):
    """
    Checks the integrity of a PyTorch checkpoint and fixes it if necessary by
    organizing it into a proper structure with 'model_state_dict' and 'optimizer_state_dict'.

    Args:
        original_checkpoint_path (str): Path to the original checkpoint file.
        fixed_checkpoint_path (str): Path where the fixed checkpoint will be saved.
        device (str): Device to map the checkpoint to ('cpu' or 'cuda').

    Returns:
        None
    """
    if not os.path.exists(original_checkpoint_path):
        logging.error(f"Checkpoint file not found: {original_checkpoint_path}")
        return

    try:
        # Load the raw checkpoint
        logging.info(f"Loading checkpoint from: {original_checkpoint_path}")
        checkpoint = torch.load(original_checkpoint_path, map_location=device)
        logging.info("Checkpoint successfully loaded!")

        # Check for the presence of 'model_state_dict' and 'optimizer_state_dict'
        has_model_state = 'model_state_dict' in checkpoint
        has_optimizer_state = 'optimizer_state_dict' in checkpoint

        if has_model_state:
            logging.info("Model state dictionary found in checkpoint.")
        else:
            logging.warning("Model state dictionary NOT found in checkpoint.")

        if has_optimizer_state:
            logging.info("Optimizer state dictionary found in checkpoint.")
        else:
            logging.warning("Optimizer state dictionary NOT found in checkpoint.")

        # If 'model_state_dict' is missing, attempt to fix the checkpoint
        if not has_model_state:
            logging.info("Attempting to fix the checkpoint by organizing it properly.")

            # Initialize the model
            model = MyModel()  # Replace with your model's initialization parameters if any
            model.to(device)

            # Handle potential DataParallel wrappers in checkpoint keys
            # If your checkpoint keys are prefixed with 'module.', remove the prefix
            state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('module.'):
                    new_key = key[len('module.'):]
                else:
                    new_key = key
                state_dict[new_key] = value

            # Load the state_dict into the model
            try:
                model.load_state_dict(state_dict, strict=True)
                logging.info("Model state_dict loaded successfully into the model.")
            except RuntimeError as e:
                logging.error(f"Error loading state_dict into the model: {e}")
                return

            # Initialize the optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust optimizer and hyperparameters as needed
            logging.info("Optimizer initialized.")

            # Optionally, load additional checkpoint information if available
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', None)

            # Create a new structured checkpoint
            new_checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
            }

            # Save the new checkpoint
            torch.save(new_checkpoint, fixed_checkpoint_path)
            logging.info(f"Fixed checkpoint saved to: {fixed_checkpoint_path}")

        else:
            logging.info("No fixing needed for the model state dictionary.")

        # Additional advice if optimizer state is missing
        if not has_optimizer_state:
            logging.warning(
                "Optimizer state is missing. When resuming training, "
                "the optimizer will be reinitialized, which might affect training dynamics."
            )

    except Exception as e:
        logging.error(f"An error occurred while processing the checkpoint: {e}")

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Check and Fix PyTorch Checkpoint Integrity")
    parser.add_argument(
        '--original_checkpoint',
        type=str,
        required=True,
        help='Path to the original checkpoint file.'
    )
    parser.add_argument(
        '--fixed_checkpoint',
        type=str,
        required=True,
        help='Path where the fixed checkpoint will be saved.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to map the checkpoint to.'
    )

    args = parser.parse_args()

    check_and_fix_checkpoint(
        original_checkpoint_path=args.original_checkpoint,
        fixed_checkpoint_path=args.fixed_checkpoint,
        device=args.device
    )

if __name__ == "__main__":
    main()