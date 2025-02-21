# my_project/utils/load_model.py

import os
import torch
import logging
from typing import Any, Dict, Optional

from my_project.models.model_stt4 import Model_STT4
from my_project.tokenizers.bioseq_tokenizer import BioSeqTokenizer


def remove_module_prefix(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove 'module.' prefixes from checkpoint keys.
    """
    return {k.replace('module.', ''): v for k, v in checkpoint.items()}


def filter_state_dict_by_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Keep only keys starting with the given prefix, removing the prefix.
    """
    prefix_length = len(prefix)
    return {k[prefix_length:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def initialize_tokenizer(args: Any) -> BioSeqTokenizer:
    """
    Initialize BioSeqTokenizer with provided arguments.
    """
    print("DEBUG: Initializing tokenizer")
    tokenizer = BioSeqTokenizer(
        window=args.window,
        stride=args.stride,
        model=args.model
    )
    args.n_token = len(tokenizer.tokens)
    args.mask_token = tokenizer.mask_token
    args.pad_token = tokenizer.pad_token
    return tokenizer


def load_checkpoint(checkpoint_path: str, device: str) -> Dict[str, Any]:
    """
    Load a checkpoint from the specified path.

    Raises:
        FileNotFoundError: If the checkpoint file is missing.
        RuntimeError: If loading fails.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    return checkpoint.get('state_dict', checkpoint)


def load_model(
    args: Any,
    checkpoint_path: Optional[str] = None,
    load_encoder_only: bool = False,
    model: Optional[Model_STT4] = None,
    strict_loading: bool = True  # New parameter to control strictness
) -> Model_STT4:
    """
    Load and initialize the Model_STT4 with configurations and checkpoint.

    This function now accepts an optional pre-initialized model (e.g., from finetune.py). If provided,
    it will update the parameters of that model from the checkpoint (if available). Otherwise, it
    instantiates a new Model_STT4.

    Args:
        args (Any): Configuration arguments.
        checkpoint_path (str, optional): Path to the checkpoint file. If None, the model is used as is.
        load_encoder_only (bool, optional): Flag to load only the encoder. Defaults to False.
        model (Model_STT4, optional): An already-initialized model instance. Defaults to None.
        strict_loading (bool, optional): If False, missing keys in the checkpoint will be allowed.
                                         Defaults to True (strict matching).

    Returns:
        Model_STT4: The updated (or newly initialized) model.

    Raises:
        ValueError: If model instantiation fails.
        FileNotFoundError: If the checkpoint file is not found.
        RuntimeError: If loading the checkpoint fails.
    """

    # If no pre-initialized model is provided, create one.
    if model is None:
        model = Model_STT4(args)
        if model is None:
            raise ValueError(f"Could not instantiate Model_STT4 with type '{args.model}'")
    else:
        logging.info("Using the pre-initialized model from finetune.py.")

    if checkpoint_path:
        # Load the checkpoint
        checkpoint = load_checkpoint(checkpoint_path, args.default)

        # Remove 'module.' prefix if present (common when using DataParallel)
        checkpoint = remove_module_prefix(checkpoint)

        if load_encoder_only:
            # -------------------------------
            # Load Only the Encoder's state_dict
            # -------------------------------
            logging.info("Loading only the encoder's state_dict from checkpoint.")
            encoder_prefix = 'encoder0.'
            encoder_state = filter_state_dict_by_prefix(checkpoint, prefix=encoder_prefix)

            # Re-add the prefix to match the model's encoder module
            encoder_state_with_prefix = {f"{encoder_prefix}{k}": v for k, v in encoder_state.items()}

            # Load the encoder's state_dict with reshaping support
            missing_keys, unexpected_keys = model.load_state_dict(
                encoder_state_with_prefix, strict=strict_loading
            )
            if missing_keys:
                logging.warning(f"Missing keys when loading encoder: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading encoder: {unexpected_keys}")

            logging.info("Encoder loaded successfully from checkpoint.")
        else:
            # -----------------------------
            # Load the Entire Model's state_dict
            # -----------------------------
            logging.info("Loading the entire model's state_dict from checkpoint.")
            checkpoint = reshape_mismatched_weights(model.state_dict(), checkpoint)

            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint, strict=strict_loading
            )
            if missing_keys:
                logging.warning(f"Missing keys when loading entire model: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading entire model: {unexpected_keys}")

            logging.info("Entire model loaded successfully from checkpoint.")
    else:
        logging.info("No checkpoint provided. Using model parameters as initialized.")

    # Move the model to the specified device
    model.to(args.default)
    return model


def reshape_mismatched_weights(model_state: Dict[str, Any], checkpoint_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reshape mismatched weights in the checkpoint to align with the model's state_dict.

    Args:
        model_state (Dict[str, Any]): The model's state_dict (expected structure).
        checkpoint_state (Dict[str, Any]): The checkpoint state_dict (loaded weights).

    Returns:
        Dict[str, Any]: Adjusted checkpoint state_dict.
    """
    adjusted_state = {}

    for key, value in checkpoint_state.items():
        if key in model_state:
            model_shape = model_state[key].shape
            if value.shape != model_shape:
                logging.warning(f"Reshaping parameter '{key}' from {value.shape} to {model_shape}.")
                try:
                    adjusted_value = value.view(model_shape)
                    adjusted_state[key] = adjusted_value
                except Exception as e:
                    logging.error(f"Failed to reshape parameter '{key}': {e}")
                    continue
            else:
                adjusted_state[key] = value
        else:
            adjusted_state[key] = value

    return adjusted_state