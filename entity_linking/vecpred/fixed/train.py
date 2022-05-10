import numpy as np
import entity_linking.vp_utils as el_util
from entity_linking.vecpred.loss import NpairLoss, NpairMSELoss
from entity_linking.vecpred.eval import ACCMetric, ACCMetricCalculator, ACC_THRESHOLDS
from entity_linking.vecpred.preprocessing import EntityIndexToVectorMapper
from entity_linking.vecpred.loss import LOSS_BCE, LOSS_MSE, LOSS_NPAIR, LOSS_NPAIRMSE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from collections import defaultdict


def train(label: str, train_loader, val_loader, entity_vectors: np.ndarray, model, lr: float, loss_type: str, epochs: int):
    # prepare transformer
    entity_index_to_vector_transformer = EntityIndexToVectorMapper(entity_vectors)
    # define loss functions and metrics
    loss_fn = {
        LOSS_MSE: nn.MSELoss(),
        LOSS_NPAIR: NpairLoss(),
        LOSS_NPAIRMSE: NpairMSELoss(),
    }
    metric_ACC = ACCMetric()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tb = SummaryWriter(log_dir=f'{el_util.get_log_path()}/{label}')
    # start training loop
    model = model.to(el_util.DEVICE)
    for t in range(epochs):
        model.train()

        running_loss = 0
        for i, (x_train, y_train) in enumerate(tqdm(train_loader, desc=f'Epoch {t}')):
            x_train = x_train.to(el_util.DEVICE)
            y_train = entity_index_to_vector_transformer(y_train.to(el_util.DEVICE))
            y_train_pos = y_train[:, 0, 0, :]  # use only vector of positive entity for training loss calculation

            y_pred = model(x_train)  # compute prediction for batch
            loss = loss_fn[loss_type](y_pred, y_train_pos)

            running_loss += loss.item()
            if i % 1000 == 999:  # log every 1000 mini-batches
                tb.add_scalar(f'{loss_type}/train', running_loss / 1000, t * len(train_loader) + i)
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
            val_metric_ACC = ACCMetricCalculator()
            for x_val, y_val in tqdm(val_loader, desc='Validation'):
                x_val = x_val.to(el_util.DEVICE)
                y_val = entity_index_to_vector_transformer(y_val.to(el_util.DEVICE))
                y_val_pos = y_val[:, 0, 0, :]
                y_pred = model(x_val)

                val_loss_MSE += loss_fn[LOSS_MSE](y_pred, y_val_pos).item()
                val_metric_ACC.add_predictions(metric_ACC(y_pred, y_val))
            tb.add_scalar(f'{LOSS_MSE}/val', val_loss_MSE / len(val_loader), t)
            val_metric_ACC.write_losses(tb, t)
    tb.close()


def train_binary(label: str, train_loader, val_loader, entity_vectors: np.ndarray, model, lr: float, loss_type: str, epochs: int):
    # prepare transformer for validation
    entity_index_to_vector_transformer = EntityIndexToVectorMapper(entity_vectors)
    # define loss functions
    loss_fn = {LOSS_BCE: nn.BCELoss()}
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tb = SummaryWriter(log_dir=f'{el_util.get_log_path()}/{label}')
    # start training loop
    model = model.to(el_util.DEVICE)
    for t in range(epochs):
        model.train()

        running_loss = 0
        for i, (x_train, y_train) in enumerate(tqdm(train_loader, desc=f'Epoch {t}')):
            x_train = x_train.to(el_util.DEVICE)
            y_train = y_train.to(el_util.DEVICE)

            y_pred = model(x_train)
            loss = loss_fn[loss_type](y_pred, y_train)

            running_loss += loss.item()
            if i % 1000 == 999:  # log every 1000 mini-batches
                tb.add_scalar(f'{loss_type}/train', running_loss / 1000, t * len(train_loader) + i)
                running_loss = 0

            # compute gradient & perform step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Epoch completed. Check accuracy over validation set
        if t % 5 != 0 and t != (epochs-1):
            continue  # only evaluate every fifth epoch
        with torch.no_grad():
            model.eval()

            entity_counts = defaultdict(int)
            correct_predictions_by_threshold = {threshold: defaultdict(int) for threshold in ACC_THRESHOLDS}
            for x_val, y_val in tqdm(val_loader, desc='Validation'):
                x_val = x_val.to(el_util.DEVICE)
                y_val = entity_index_to_vector_transformer(y_val.to(el_util.DEVICE))
                # evaluate every example in the batch individually
                for x, (entity_vecs, entity_mask) in zip(x_val, y_val):
                    # entity is valid if respective first element is 1, invalid if 0
                    entity_mask = entity_mask[:, 0].bool()

                    pos_ent_vector = entity_vecs[0]
                    is_known_entity = entity_mask[0].item()

                    neg_ent_vectors = entity_vecs[1:][entity_mask[1:]]
                    has_negatives = len(neg_ent_vectors) > 0

                    entity_counts[(is_known_entity, has_negatives)] += 1

                    if has_negatives:
                        # if there are negatives, prediction is incorrect if any negative is predicted as match
                        negative_features = torch.cat((x.repeat(neg_ent_vectors.shape[0], 1), neg_ent_vectors), dim=-1)
                        prediction_negatives = model(negative_features)
                        if (prediction_negatives >= .5).any():
                            continue

                    if is_known_entity:
                        # if entity is known, prediction is incorrect if positive instance is not predicted as match
                        prediction_positive = model(torch.cat((x, pos_ent_vector)).unsqueeze(0)).item()
                        if prediction_positive < .5:
                            continue

                    # otherwise, prediction is correct
                    for threshold in ACC_THRESHOLDS:
                        correct_predictions_by_threshold[threshold][(is_known_entity, has_negatives)] += 1

            val_metric_ACC = ACCMetricCalculator()
            val_metric_ACC.add_predictions((entity_counts, correct_predictions_by_threshold))
            val_metric_ACC.write_losses(tb, t)
    tb.close()
