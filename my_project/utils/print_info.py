# my_project/utils/print_info.py

import logging

def print_parameters(config):
    """
    Logs the configuration parameters.
    """
    logging.info("=== Configuration Parameters ===")
    for key, value in vars(config).items():
        logging.info(f"{key}: {value}")
    logging.info("=== End of Configuration Parameters ===\n")

def print_model_architecture(model):
    """
    Logs the model architecture.
    """
    logging.info("=== Model Architecture ===")
    logging.info(model)
    logging.info("=== End of Model Architecture ===\n")