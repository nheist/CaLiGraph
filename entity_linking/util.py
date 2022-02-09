import torch
import numpy as np
import pandas as pd
import pickle
import random
from pathlib import Path

# defs
EPS = 1e-8
MAX_CORES = 12
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_PARTS = 10
DATASET_VALIDATION_SIZE = .5  # portion of chunk 0 that will be used as validation set

# set seed
SEED = 41
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# paths
DATASET_PATH = './data_disambiguation'
DATASET_ID = 'clgv21-v1'
TORCH_LOG_DIR = f'{DATASET_PATH}/logs/{DATASET_ID}'

# target embeddings
TARGET_EMBEDDING_PATH = f'{DATASET_PATH}/embeddings/dbpedia2020_vectors.txt'
TARGET_EMBEDDING_DIMENSIONS = 200


def store_data(data, filename: str, parts=None):
    filepath = _get_filepath(filename, parts)
    if filename.endswith('.p'):
        with open(filepath, mode='wb') as f:
            pickle.dump(data, f)
    elif filename.endswith('.feather'):
        data.to_feather(filepath)
    else:
        raise ValueError(f'Unknown file extension for filename: {filename}')


def load_data(filename: str, parts=None):
    filepath = _get_filepath(filename, parts)
    if filename.endswith('.p'):
        with open(filepath, mode='rb') as f:
            return pickle.load(f)
    elif filename.endswith('.feather'):
        return pd.read_feather(filepath)
    else:
        raise ValueError(f'Unknown file extension for filename: {filename}')


def file_exists(filename: str, parts=None):
    filepath = _get_filepath(filename, parts)
    return Path(filepath).exists()


def _get_filepath(filename: str, parts) -> str:
    if parts:
        return f'{DATASET_PATH}/{DATASET_ID}_{parts}p_{filename}'
    else:
        return f'{DATASET_PATH}/{DATASET_ID}_{filename}'
