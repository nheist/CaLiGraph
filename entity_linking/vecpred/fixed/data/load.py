import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, Sampler
from typing import Iterator
import entity_linking.util as el_util
from typing import Tuple


def get_entity_vectors(parts: int) -> Tuple[np.ndarray, dict, dict]:
    entity_vectors = el_util.load_data('entity-vectors.feather', parts=parts)
    idx2ent, ent2idx = _get_entity_vector_indices(entity_vectors)
    return entity_vectors.drop(columns='ent').to_numpy(), idx2ent, ent2idx


def _get_entity_vector_indices(entity_vectors: pd.DataFrame) -> Tuple[dict, dict]:
    idx2ent = entity_vectors['ent'].to_dict()
    ent2idx = {ent: idx for idx, ent in idx2ent.items()}
    return idx2ent, ent2idx


def get_train_and_val_data(parts: int, ent2idx: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = el_util.load_data('X.feather', parts=parts)
    Y = el_util.load_data('Y.p', parts=parts)

    # train/validation split
    val_pages = el_util.load_data('validation-set-pages.p')
    val_mask = X['_page'].isin(val_pages)
    X_train, Y_train = X[~val_mask], Y[~val_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]

    # finalise datasets
    known_entity_mask = X_train['_ent'].isin(set(ent2idx))
    X_train = X_train[known_entity_mask].reset_index(drop=True)
    Y_train = Y_train[known_entity_mask]
    context_features = ['_ent', '_page', '_text', '_section']
    X_train = X_train.drop(columns=context_features).to_numpy()
    X_val = X_val.drop(columns=context_features).to_numpy()

    return X_train, Y_train, X_val, Y_val


def get_data_loader(X: np.ndarray, Y: np.ndarray, ent2idx: dict, batch_size=32, hard_negative_blocks=None, ignore_singles=False):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).int())
    if hard_negative_blocks:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=HardNegativeSampler(Y, ent2idx, hard_negative_blocks, ignore_singles))
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader


class HardNegativeSampler(Sampler[int]):
    """Samples elements using a loose order induced by entities with similar surface forms."""

    def __init__(self, Y: np.ndarray, ent2idx: dict, hard_negative_blocks: dict, ignore_singles: bool):
        super().__init__(None)
        self.row_grps = self._group_similar_entities(Y, ent2idx, hard_negative_blocks, ignore_singles)
        self.data_size = sum(len(grp) for grp in self.row_grps)

    def _group_similar_entities(self, Y: np.ndarray, ent2idx: dict, hard_negative_blocks: dict, ignore_singles: bool):
        # create mapping from example index to entity id
        train_entity_indices = {e_id: idx for idx, e_id in enumerate(Y[:, 0])}
        # create groups of rows with disjoint entities from surface form blocks
        row_groups = []
        grouped_rows = set()
        for ent_block in hard_negative_blocks.values():
            ent_ids = {ent2idx[e] for e in ent_block if e in ent2idx}  # convert to entity id
            row_ids = {train_entity_indices[e] for e in ent_ids if e in train_entity_indices}  # convert to row id
            row_ids = {i for i in row_ids if i not in grouped_rows}  # remove already grouped rows
            if row_ids and (not ignore_singles or len(row_ids) > 1):
                row_groups.append(row_ids)
                grouped_rows.update(row_ids)
        if not ignore_singles:
            # add missing rows as singular groups
            for i in range(len(train_entity_indices)):
                if i not in grouped_rows:
                    row_groups.append({i})
        return row_groups

    def __iter__(self) -> Iterator[int]:
        return iter(idx for grp in random.sample(self.row_grps, len(self.row_grps)) for idx in grp)

    def __len__(self) -> int:
        return self.data_size
