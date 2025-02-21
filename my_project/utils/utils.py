import logging
import pandas as pd
import torch
import random
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataframe(path):
    """
    Load a dataframe from the given path.
    Supports CSV, TSV, and other common formats.
    """
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.tsv'):
        return pd.read_csv(path, sep='\t')
    else:
        raise ValueError(f"Unsupported file format for path: {path}")