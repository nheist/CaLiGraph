import torch
import numpy as np
import pandas as pd
import pickle
import random
from pathlib import Path
import impl.dbpedia.util as dbp_util

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
ROOT_PATH = './data_disambiguation'
DATA_PATH = f'{ROOT_PATH}/data'
DATASET_ID = 'clgv21-v1'
LOG_PATH = f'{ROOT_PATH}/logs/{DATASET_ID}'

# target embeddings
EMBEDDING_TYPE_SIZES = {'rdf2vec': 200}


def load_entity_vectors(embedding_type: str):
    path = f'{ROOT_PATH}/embeddings/{embedding_type}_vectors.txt'
    size = EMBEDDING_TYPE_SIZES[embedding_type]

    vecs = pd.read_csv(path, sep=' ', usecols=list(range(size + 1)), names=['ent'] + [f'v_{i}' for i in range(size)])
    vecs = vecs[vecs['ent'].str.startswith(dbp_util.NAMESPACE_DBP_RESOURCE)]
    vecs['ent'] = vecs['ent'].transform(dbp_util.resource2name)
    return vecs


def store_data(data, filename: str, parts=None, embedding_type=None):
    filepath = _get_filepath(filename, parts, embedding_type)
    if filename.endswith('.p'):
        with open(filepath, mode='wb') as f:
            pickle.dump(data, f)
    elif filename.endswith('.feather'):
        data.to_feather(filepath)
    else:
        raise ValueError(f'Unknown file extension for filename: {filename}')


def load_data(filename: str, parts=None, embedding_type=None):
    filepath = _get_filepath(filename, parts, embedding_type)
    if filename.endswith('.p'):
        with open(filepath, mode='rb') as f:
            return pickle.load(f)
    elif filename.endswith('.feather'):
        return pd.read_feather(filepath)
    else:
        raise ValueError(f'Unknown file extension for filename: {filename}')


def file_exists(filename: str, parts=None, embedding_type=None):
    filepath = _get_filepath(filename, parts, embedding_type)
    return Path(filepath).exists()


def _get_filepath(filename: str, parts: int, embedding_type: str) -> str:
    filepath = f'{DATA_PATH}/{DATASET_ID}'
    if parts:
        filepath += f'_{parts}p'
    if embedding_type:
        filepath += f'_{embedding_type}'
    filepath += f'_{filename}'
    return filepath
