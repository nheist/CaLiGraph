import numpy as np
import pandas as pd
import random
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader, Sampler
from typing import Iterator
import entity_linking.utils as el_util
from typing import Tuple


def get_entity_vectors(parts: int, embedding_type: str) -> Tuple[np.ndarray, dict, dict]:
    entity_vectors = el_util.load_data('entity-vectors.feather', parts=parts, embedding_type=embedding_type)
    idx2ent, ent2idx = _get_entity_vector_indices(entity_vectors)
    return entity_vectors.drop(columns='ent').to_numpy(), idx2ent, ent2idx


def _get_entity_vector_indices(entity_vectors: pd.DataFrame) -> Tuple[dict, dict]:
    idx2ent = entity_vectors['ent'].to_dict()
    ent2idx = {ent: idx for idx, ent in idx2ent.items()}
    return idx2ent, ent2idx


def get_train_and_val_data(parts: int, embedding_type: str, ent2idx: dict, remove_unknown_entities: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = el_util.load_data('X.feather', parts=parts, embedding_type=embedding_type)
    Y = el_util.load_data('Y.p', parts=parts, embedding_type=embedding_type)

    # train/validation split
    val_pages = el_util.load_data('validation-set-pages.p')
    val_mask = X['_page'].isin(val_pages)
    X_train, Y_train = X[~val_mask], Y[~val_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]

    # finalise datasets
    if remove_unknown_entities:
        known_entity_mask = X_train['_ent'].isin(set(ent2idx))
        X_train = X_train[known_entity_mask].reset_index(drop=True)
        Y_train = Y_train[known_entity_mask]
    context_features = ['_ent', '_page', '_text', '_section']
    X_train = X_train.drop(columns=context_features).to_numpy()
    X_val = X_val.drop(columns=context_features).to_numpy()

    return X_train, Y_train, X_val, Y_val


def get_data_loader(X: np.ndarray, Y: np.ndarray, batch_size=32):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).int())
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def get_data_loader_with_hard_negatives(X: np.ndarray, Y: np.ndarray, ent2idx: dict, hard_negative_blocks: dict, batch_size=32):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).int())
    sampler = HardNegativeSampler(Y, ent2idx, hard_negative_blocks)
    return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)


def get_data_loader_for_binary_matching(X: np.ndarray, Y: np.ndarray, entity_vectors: np.ndarray, batch_size=32):
    dataset = WithNegativesDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).int(), entity_vectors)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


class HardNegativeSampler(Sampler[int]):
    """Samples elements using a loose order induced by entities with similar surface forms."""
    def __init__(self, Y: np.ndarray, ent2idx: dict, hard_negative_blocks: dict):
        super().__init__(None)
        self.row_grps = self._group_similar_entities(Y, ent2idx, hard_negative_blocks)
        self.data_size = sum(len(grp) for grp in self.row_grps)

    def _group_similar_entities(self, Y: np.ndarray, ent2idx: dict, hard_negative_blocks: dict):
        # create mapping from example index to entity id
        train_entity_indices = {e_id: idx for idx, e_id in enumerate(Y[:, 0])}
        # create groups of rows with disjoint entities from surface form blocks
        row_groups = []
        grouped_rows = set()
        for ent_block in hard_negative_blocks.values():
            ent_ids = {ent2idx[e] for e in ent_block if e in ent2idx}  # convert to entity id
            row_ids = {train_entity_indices[e] for e in ent_ids if e in train_entity_indices}  # convert to row id
            row_ids = {i for i in row_ids if i not in grouped_rows}  # remove already grouped rows
            if row_ids:
                row_groups.append(row_ids)
                grouped_rows.update(row_ids)
        # add missing rows as singular groups
        for i in range(len(train_entity_indices)):
            if i not in grouped_rows:
                row_groups.append({i})
        return row_groups

    def __iter__(self) -> Iterator[int]:
        return iter(idx for grp in random.sample(self.row_grps, len(self.row_grps)) for idx in grp)

    def __len__(self) -> int:
        return self.data_size


class WithNegativesDataset(Dataset):
    def __init__(self, X: Tensor, Y: Tensor, entity_vectors: np.array):
        self.X = X
        self.entity_vectors = torch.from_numpy(entity_vectors).float()
        self.data_indices, self.labels = self._index_data(Y)

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        x_idx, e_idx = self.data_indices[idx]
        return torch.cat((self.X[x_idx], self.entity_vectors[e_idx])), self.labels[idx]

    def _index_data(self, Y: Tensor):
        # only vectors that do not point to -1 are valid
        valid_indices = Y.flatten() != -1
        # flatten Y and add the index of the first dimension as additional value
        # e.g. [[0,1], [2,3]] is transformed to [[0,0], [0,1], [1,2], [1,3]]
        dim0_indices = torch.arange(len(Y)).unsqueeze(-1).expand_as(Y)
        flattened_Y = torch.cat((dim0_indices.unsqueeze(-1), Y.unsqueeze(-1)), -1).flatten(end_dim=-2)
        # finally, filter on valid vector indices
        valid_Y = flattened_Y[valid_indices]

        # always marks the first vector as correct and all the others as incorrect
        all_labels = torch.zeros_like(Y)
        all_labels[:, 0] = 1
        # finally, filter on valid vectors
        valid_labels = all_labels.flatten()[valid_indices].unsqueeze(-1).float()

        return valid_Y, valid_labels
