import argparse
import numpy as np
import pandas as pd
import torch
import random
import pickle
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, Sampler
import torch.nn as nn
from torch import Tensor
from typing import Iterator
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

# definitions

EPS = 1e-10
SEED = 41
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_PATH = './data_disambiguation'
DATASET_ID = 'clgv21-v1'

TORCH_LOG_DIR = f'{DATASET_PATH}/logs/{DATASET_ID}'


FEATURES_FULL = 'full'

# loss strategies used for training
LOSS_MSE = 'MSE'
LOSS_COS = 'COS'
LOSS_NPAIR = 'NPAIR'
LOSS_NPAIRCOS = 'NPAIR+COS'
LOSS_NPAIRMSE = 'NPAIR+MSE'

LOSS_MSENPAIR = 'MSE-NPAIR'
LOSS_MSENPAIR_ALT = 'MSE-NPAIR-alt'
LOSS_COSNPAIR = 'COS-NPAIR'
LOSS_COSNPAIR_ALT = 'COS-NPAIR-alt'

LOSS_TYPES = [LOSS_COS, LOSS_MSE, LOSS_NPAIR, LOSS_NPAIRCOS, LOSS_NPAIRMSE, LOSS_MSENPAIR, LOSS_MSENPAIR_ALT, LOSS_COSNPAIR, LOSS_COSNPAIR_ALT]

# losses used only for evaluation
LOSS_MSEACC = 'MSEACC'
LOSS_COSACC = 'COSACC'


ACC_MSE_THRESHOLDS = [999, .5, .4, .35, .3, .2]
ACC_COS_THRESHOLDS = [999, .5, .4, .35, .3, .2]


# data loading utils

def load_data(filename: str, parts=None):
    filepath = _get_filepath(filename, parts)
    if filename.endswith('.p'):
        with open(filepath, mode='rb') as f:
            return pickle.load(f)
    elif filename.endswith('.feather'):
        return pd.read_feather(filepath)
    else:
        raise ValueError(f'Unknown file extension for filename: {filename}')


def _get_filepath(filename: str, parts) -> str:
    if parts:
        return f'{DATASET_PATH}/{DATASET_ID}_{parts}p_{filename}'
    else:
        return f'{DATASET_PATH}/{DATASET_ID}_{filename}'


# data loading & sampling

def get_data_loader(X: pd.DataFrame, Y: np.ndarray, ent2idx: dict, batch_size=32, hard_negative_blocks=None, ignore_singles=False):
    dataset = TensorDataset(torch.from_numpy(X.values).float(), torch.Tensor(Y).int())
    if hard_negative_blocks:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                                 sampler=HardNegativeSampler(Y, ent2idx, hard_negative_blocks, ignore_singles))
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


# data transformation

class EntityIndexToVectorTranformer(nn.Module):
    """
    Takes a vector of (batch_size, entity_count) containing entity indices
    and transforms it into (batch_size, 2, entity_count, vector_dim)
    where the first array of size (entity_count, vector_dim) contains the entity vectors
    and the second contains a validity mask specifying whether the vectors should be used or not
    """

    def __init__(self, entity_vectors):
        super().__init__()
        self.entity_vectors = torch.from_numpy(entity_vectors).float().to(device)
        self.vector_dim = self.entity_vectors.shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        # create mask that indicates whether a vector is valid or not
        mask = (x != -1)
        # flatten input indices, retrieve vectors, reshape to (batch_size, entity_count, vector_dim)
        flattened_indices = x.reshape(-1)
        flattened_indices[
            flattened_indices == -1] = 0  # set invalid indices to 0 (torch.index_select throws error for negative values)
        flattened_vecs = torch.index_select(self.entity_vectors, 0, flattened_indices)
        batch_size, entity_count = x.shape
        vecs = flattened_vecs.reshape((batch_size, entity_count, self.vector_dim))
        mask = mask.unsqueeze(-1).expand_as(vecs).float()  # adapt to layout of vecs
        # stack to final shape of (batch_size, 2, entity_count, vector_dim)
        return torch.stack((vecs, mask), dim=1)


# losses

class Dim1MSELoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        reduce_dim = input.dim() - 1
        return torch.square(torch.subtract(input, target)).mean(dim=reduce_dim)


class COSLoss(nn.Module):
    def __init__(self, reduce=True):
        super().__init__()
        self.distance_fn = nn.CosineSimilarity()
        self.reduce = reduce

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # add dummy dimension if not used in batch mode
        if input.dim() == 1:
            input = input.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        # transform from *similarity* in [-1,1] to *distance* in [0,1]
        loss = (1 - self.distance_fn(input, target)) / 2
        return loss.mean() if self.reduce else loss


class NpairLoss(nn.Module):
    """
    Computes multi-class N-pair loss according to
    http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
    but assumes that the labels of all entities in input are different.

    Adapted from https://github.com/ChaofWang/Npair_loss_pytorch/blob/master/Npair_loss.py
    """

    def __init__(self, l2_reg=0.02):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.l2_reg = l2_reg

    def forward(self, input, target):
        batch_size = input.size(0)
        labels = torch.arange(batch_size).to(device)
        logit = torch.matmul(input, torch.transpose(target, 0, 1))
        loss_ce = self.cross_entropy(logit, labels)

        l2_loss = torch.sum(input ** 2) / batch_size + torch.sum(target ** 2) / batch_size

        return loss_ce + self.l2_reg * l2_loss * 0.25


class NpairCOSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.npair_loss = NpairLoss()
        self.cosine_loss = COSLoss()

    def forward(self, input, target):
        return self.npair_loss(input, target) + self.cosine_loss(input, target)


class NpairMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.npair_loss = NpairLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        return self.npair_loss(input, target) + self.mse_loss(input, target)


class ACCLoss(nn.Module):
    NO_THRESHOLD = 999

    def __init__(self, distance_fn, distance_thresholds):
        super().__init__()
        self.distance_fn = distance_fn
        self.distance_thresholds = distance_thresholds

    def forward(self, input: Tensor, target: Tensor):
        entity_counts = defaultdict(int)
        incorrect_predictions_by_threshold = {t: defaultdict(int) for t in self.distance_thresholds}
        for predicted_vector, (entity_vectors, entity_mask) in zip(input, target):  # iterate over input batch
            entity_mask = entity_mask[:, 0].bool()  # entity is valid if respective first element is 1, invalid if 0
            is_known_entity = entity_mask[0].item()

            neg_ent_vectors = entity_vectors[entity_mask][1:]
            has_negatives = len(neg_ent_vectors) > 0

            entity_counts[(is_known_entity, has_negatives)] += 1

            if not is_known_entity and not has_negatives:
                continue  # any prediction would be correct in this case -> disregard

            pos_ent_vector = entity_vectors[0]
            pos_distance = self.distance_fn(pos_ent_vector, predicted_vector).item()

            if not has_negatives:
                # if no negatives, prediction is only incorrect if distance to positive is greather than threshold
                for t in self.distance_thresholds:
                    if t < pos_distance:
                        incorrect_predictions_by_threshold[t][(is_known_entity, has_negatives)] += 1
                continue

            neg_distances = self.distance_fn(neg_ent_vectors, predicted_vector)
            min_neg_distance = torch.min(neg_distances).item()
            if not is_known_entity:
                # if no positive, prediction is incorrect if any negative is within the threshold
                for t in self.distance_thresholds:
                    if min_neg_distance < t:
                        incorrect_predictions_by_threshold[t][(is_known_entity, has_negatives)] += 1
                continue

            # otherwise, prediction is incorrect if distance to positive is greater than to any negative or threshold
            for t in self.distance_thresholds:
                min_distance = min(min_neg_distance, t)
                if min_distance < pos_distance:
                    incorrect_predictions_by_threshold[t][(is_known_entity, has_negatives)] += 1
        return entity_counts, incorrect_predictions_by_threshold


class ACCLossCalculator:
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.entity_counts = defaultdict(int)
        self.incorrect_predictions_by_threshold = defaultdict(lambda: defaultdict(int))

    def add_predictions(self, prediction_tuple):
        entity_counts, incorrect_predictions_by_threshold = prediction_tuple
        for status, val in entity_counts.items():
            self.entity_counts[status] += val
        for t, ip in incorrect_predictions_by_threshold.items():
            for status, val in ip.items():
                self.incorrect_predictions_by_threshold[t][status] += val

    def write_losses(self, tb: SummaryWriter, epoch: int):
        ents_total = sum(self.entity_counts.values())
        ents_known_negatives = self.entity_counts[(True, True)]
        ents_known_nonegatives = self.entity_counts[(True, False)]
        ents_unknown_negatives = self.entity_counts[(False, True)]
        ents_unknown_nonegatives = self.entity_counts[(False, False)]

        for t, ip in self.incorrect_predictions_by_threshold.items():
            ip_total = sum(ip.values())
            ip_known_negatives = ip[(True, True)]
            ip_known_nonegatives = ip[(True, False)]
            ip_unknown_negatives = ip[(False, True)]
            ip_unknown_nonegatives = ip[(False, False)]

            acc_total = ip_total / ents_total
            acc_known = (ip_known_negatives + ip_known_nonegatives) / (
                        ents_known_negatives + ents_known_nonegatives + EPS)
            acc_unknown = (ip_unknown_negatives + ip_unknown_nonegatives) / (
                        ents_unknown_negatives + ents_unknown_nonegatives + EPS)
            acc_negatives = (ip_known_negatives + ip_unknown_negatives) / (
                        ents_known_negatives + ents_unknown_negatives + EPS)
            acc_nonegatives = (ip_known_nonegatives + ip_unknown_nonegatives) / (
                        ents_known_nonegatives + ents_unknown_nonegatives + EPS)
            acc_known_negatives = ip_known_negatives / (ents_known_negatives + EPS)
            acc_known_nonegatives = ip_known_nonegatives / (ents_known_nonegatives + EPS)
            acc_unknown_negatives = ip_unknown_negatives / (ents_unknown_negatives + EPS)

            tb.add_scalar(self._get_loss_name('all', t), acc_total, epoch)
            tb.add_scalar(self._get_loss_name('known', t), acc_known, epoch)
            tb.add_scalar(self._get_loss_name('unknown', t), acc_unknown, epoch)
            tb.add_scalar(self._get_loss_name('negatives', t), acc_negatives, epoch)
            tb.add_scalar(self._get_loss_name('nonegatives', t), acc_nonegatives, epoch)
            tb.add_scalar(self._get_loss_name('known_negatives', t), acc_known_negatives, epoch)
            tb.add_scalar(self._get_loss_name('unknown_negatives', t), acc_unknown_negatives, epoch)
            tb.add_scalar(self._get_loss_name('known_nonegatives', t), acc_known_nonegatives, epoch)

    def _get_loss_name(self, loss_type: str, threshold: float):
        threshold = 0 if threshold == ACCLoss.NO_THRESHOLD else threshold
        return f'ACC-{self.name}-T{threshold}/{loss_type}'


# training

def train(parts, train_loader, val_loader, entity_vectors, model, lr, loss_type, features=FEATURES_FULL, epochs=100, prefix=''):
    prefix = prefix + '_' if prefix else prefix
    log_name = f'{TORCH_LOG_DIR}/{prefix}{parts}P_{type(model).__name__}_feat-{features}_lr-{lr}_loss-{loss_type}'

    # prepare transformer
    entity_index_to_vector_transformer = EntityIndexToVectorTranformer(entity_vectors)

    # define loss functions
    loss_fn = {
        LOSS_MSE: nn.MSELoss(),
        LOSS_COS: COSLoss(),
        LOSS_NPAIR: NpairLoss(),
        LOSS_NPAIRCOS: NpairCOSLoss(),
        LOSS_NPAIRMSE: NpairMSELoss(),
        LOSS_MSEACC: ACCLoss(Dim1MSELoss(), ACC_MSE_THRESHOLDS),
        LOSS_COSACC: ACCLoss(COSLoss(reduce=False), ACC_COS_THRESHOLDS),
    }

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tb = SummaryWriter(log_dir=log_name)
    # start training loop
    model = model.to(device)
    for t in range(epochs):
        model.train()

        loss_name = _resolve_loss_strategy(loss_type, t, epochs)
        running_loss = 0
        for i, (x_train, y_train) in enumerate(tqdm(train_loader, desc=f'Epoch {t}')):
            x_train = x_train.to(device)
            y_train = entity_index_to_vector_transformer(y_train.to(device))
            y_train_pos = y_train[:, 0, 0, :]  # use only vector of positive entity for training loss calculation

            y_pred = model(x_train)  # compute prediction for batch
            loss = loss_fn[loss_name](y_pred, y_train_pos)

            running_loss += loss.item()
            if i % 1000 == 999:  # log every 1000 mini-batches
                tb.add_scalar(f'{loss_name}/train', running_loss / 1000, t * len(train_loader) + i)
                running_loss = 0

            # compute gradient & perform step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Epoch completed. Check accuracy over validation set
        if t % 10 != 0 and t != (epochs - 1):
            continue  # only evaluate every tenth epoch
        with torch.no_grad():
            model.eval()
            val_loss_MSE = 0
            val_loss_COS = 0
            val_loss_MSEACC = ACCLossCalculator(LOSS_MSE)
            val_loss_COSACC = ACCLossCalculator(LOSS_COS)
            for x_val, y_val in tqdm(val_loader, desc='Validation'):
                x_val = x_val.to(device)
                y_val = entity_index_to_vector_transformer(y_val.to(device))
                y_val_pos = y_val[:, 0, 0, :]
                y_pred = model(x_val)

                val_loss_MSE += loss_fn[LOSS_MSE](y_pred, y_val_pos).item()
                val_loss_COS += loss_fn[LOSS_COS](y_pred, y_val_pos).item()
                val_loss_MSEACC.add_predictions(loss_fn[LOSS_MSEACC](y_pred, y_val))
                val_loss_COSACC.add_predictions(loss_fn[LOSS_COSACC](y_pred, y_val))
            tb.add_scalar('MSE/val', val_loss_MSE / len(val_loader), t)
            tb.add_scalar('COS/val', val_loss_COS / len(val_loader), t)
            val_loss_MSEACC.write_losses(tb, t)
            val_loss_COSACC.write_losses(tb, t)
    tb.close()


def _resolve_loss_strategy(loss_type: str, epoch: int, total_epochs: int) -> str:
    if loss_type in {LOSS_MSE, LOSS_COS, LOSS_NPAIR}:
        return loss_type
    if loss_type in {LOSS_MSENPAIR, LOSS_COSNPAIR} and epoch >= (total_epochs / 2):
        return LOSS_NPAIR
    if loss_type in {LOSS_MSENPAIR_ALT, LOSS_COSNPAIR_ALT} and epoch % 2 != 0:
        return LOSS_NPAIR
    return LOSS_COS if loss_type.startswith(LOSS_COS) else LOSS_MSE


# models

class FCNN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        # Learnable layers
        self.linear1 = nn.Linear(n_features, n_features)
        self.linear2 = nn.Linear(n_features, 200)  # predict 200-dim RDF2vec vector

    def forward(self, x):
        # x.size() = (N, n_features)
        x = F.relu(self.linear1(x))
        # x.size() = (N, n_features)
        x = self.linear2(x)
        # x.size() = (N, 200)
        return x


class FCNN2(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        # Learnable layers
        self.linear1 = nn.Linear(n_features, n_features)
        self.linear2 = nn.Linear(n_features, n_features)
        self.linear3 = nn.Linear(n_features, 200)  # predict 200-dim RDF2vec vector

    def forward(self, x):
        # x.size() = (N, n_features)
        x = F.relu(self.linear1(x))
        # x.size() = (N, n_features)
        x = F.relu(self.linear2(x))
        # x.size() = (N, n_features)
        x = self.linear3(x)
        # x.size() = (N, 200)
        return x


class FCNN3(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        # Learnable layers
        self.linear1 = nn.Linear(n_features, n_features)
        self.linear2 = nn.Linear(n_features, n_features)
        self.linear3 = nn.Linear(n_features, n_features)
        self.linear4 = nn.Linear(n_features, 200)  # predict 200-dim RDF2vec vector

    def forward(self, x):
        # x.size() = (N, n_features)
        x = F.relu(self.linear1(x))
        # x.size() = (N, n_features)
        x = F.relu(self.linear2(x))
        # x.size() = (N, n_features)
        x = F.relu(self.linear3(x))
        # x.size() = (N, n_features)
        x = self.linear4(x)
        # x.size() = (N, 200)
        return x


class FCNN4(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        # Learnable layers
        self.linear1 = nn.Linear(n_features, n_features)
        self.linear2 = nn.Linear(n_features, n_features)
        self.linear3 = nn.Linear(n_features, n_features)
        self.linear4 = nn.Linear(n_features, n_features)
        self.linear5 = nn.Linear(n_features, 200)  # predict 200-dim RDF2vec vector

    def forward(self, x):
        # x.size() = (N, n_features)
        x = F.relu(self.linear1(x))
        # x.size() = (N, n_features)
        x = F.relu(self.linear2(x))
        # x.size() = (N, n_features)
        x = F.relu(self.linear3(x))
        # x.size() = (N, n_features)
        x = F.relu(self.linear4(x))
        # x.size() = (N, n_features)
        x = self.linear5(x)
        # x.size() = (N, 200)
        return x


# run evaluation


def run_evaluation(parts: int, loss: str, epochs: int, batch_size: int, hard_negatives: bool, ignore_singles: bool, prefix: str):
    # retrieve X and Y
    print('Loading X and Y..')
    X = load_data('X.feather', parts=parts)
    Y = load_data('Y.p', parts=parts)

    # train/validation split
    print('Splitting train and validation set..')
    val_pages = load_data('validation-set-pages.p')
    val_mask = X['_page'].isin(val_pages)
    X_train, Y_train = X[~val_mask], Y[~val_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]

    # retrieve entity vectors
    print('Retrieving entity vectors..')
    entity_vectors = load_data('entity-vectors.feather', parts=parts)
    idx2ent = entity_vectors['ent'].to_dict()
    ent2idx = {ent: idx for idx, ent in idx2ent.items()}
    entity_vectors = entity_vectors.drop(columns='ent').to_numpy()

    # finalise datasets
    print('Finalising datasets..')
    known_entity_mask = X_train['_ent'].isin(set(ent2idx))
    X_train = X_train[known_entity_mask].reset_index(drop=True)
    Y_train = Y_train[known_entity_mask]
    context_features = ['_ent', '_page', '_text', '_section']
    X_train = X_train.drop(columns=context_features)
    X_val = X_val.drop(columns=context_features)

    # retrieve entity blocks
    print('Retrieving entity blocks..')
    sf_to_entity_word_mapping = load_data('sf-to-entity-word-mapping.p', parts=parts)

    # prepare data loaders
    print('Preparing data loaders..')
    if hard_negatives:
        train_loader = get_data_loader(X_train, Y_train, ent2idx, batch_size=batch_size, hard_negative_blocks=sf_to_entity_word_mapping, ignore_singles=ignore_singles)
    else:
        train_loader = get_data_loader(X_train, Y_train, ent2idx, batch_size=batch_size)
    val_loader = get_data_loader(X_val, Y_val, ent2idx)
    n_features = len(X_train.columns)

    # run training
    for model in [FCNN2(n_features), FCNN4(n_features)]:
        print(f'Running training for model {type(model).__name__}..')
        train(parts, train_loader, val_loader, entity_vectors, model, 1e-5, loss, epochs=epochs, prefix=prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the evaluation of RDF2vec vector prediction on DBpedia.')
    parser.add_argument('parts', type=int, choices=list(range(1, 11)), help='specify how many parts of the dataset (max: 10) are used')
    parser.add_argument('loss', choices=LOSS_TYPES, help='loss function used for training')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='size of batches for training')
    parser.add_argument('-p', '--prefix', default='', help='additional prefix in the name of the model')
    parser.add_argument('-n', '--hard_negatives', action="store_true", help='put similar examples together in the same batch')
    parser.add_argument('-i', '--ignore_singles', action="store_true", help='ignore examples without similar entities')
    args = parser.parse_args()

    prefix = args.prefix
    if args.hard_negatives:
        prefix = prefix + 'HN'
    if args.ignore_singles:
        prefix = prefix + 'IS'
    run_evaluation(args.parts, args.loss, args.epochs, args.batch_size, args.hard_negatives, args.ignore_singles, prefix)
