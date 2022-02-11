from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
from torch.utils.tensorboard import SummaryWriter
import entity_linking.util as el_util
from entity_linking.vecpred.eval import ACCMetric, ACCMetricCalculator, ACC_THRESHOLDS
from entity_linking.vecpred.preprocessing import EntityIndexToVectorMapper


def evaluate_baselines(val_loader: DataLoader, entity_vectors: np.ndarray, idx2ent: dict, epochs: int):
    # prepare transformer
    entity_index_to_vector_transformer = EntityIndexToVectorMapper(entity_vectors)
    with torch.no_grad():
        _baseline_random(val_loader, entity_index_to_vector_transformer, epochs)
        _baseline_mean(val_loader, entity_index_to_vector_transformer, epochs)
        _baseline_popularity(val_loader, idx2ent, epochs)


def _baseline_random(val_loader: DataLoader, entity_index_to_vector_transformer: nn.Module, epochs: int):
    baseline_name = 'BASELINE_RANDOM'
    tb = SummaryWriter(log_dir=f'{el_util.LOG_PATH}/{baseline_name}')

    metric_ACC = ACCMetric()
    calculator_ACC = ACCMetricCalculator()

    for _, y_val in tqdm(val_loader, desc=baseline_name):
        y_val = entity_index_to_vector_transformer(y_val.to(el_util.DEVICE))
        pred_shape = (y_val.shape[0], y_val.shape[-1])
        y_pred = (torch.rand(pred_shape) * 2 - 1).to(el_util.DEVICE)  # produce random predictions in [-1, 1]
        calculator_ACC.add_predictions(metric_ACC(y_pred, y_val))

    for t in range(epochs):
        calculator_ACC.write_losses(tb, t)
    tb.close()


def _baseline_mean(val_loader: DataLoader, entity_index_to_vector_transformer: nn.Module, epochs: int):
    baseline_name = 'BASELINE_MEAN'
    tb = SummaryWriter(log_dir=f'{el_util.LOG_PATH}/{baseline_name}')

    # predict the mean of all entity vectors
    y_pred_vector = entity_index_to_vector_transformer.entity_vectors.mean(0).to(el_util.DEVICE)

    metric_ACC = ACCMetric()
    calculator_ACC = ACCMetricCalculator()

    for _, y_val in tqdm(val_loader, desc=baseline_name):
        y_val = entity_index_to_vector_transformer(y_val.to(el_util.DEVICE))
        y_pred = y_pred_vector.expand(y_val.shape[0], -1)
        calculator_ACC.add_predictions(metric_ACC(y_pred, y_val))

    for t in range(epochs):
        calculator_ACC.write_losses(tb, t)
    tb.close()


def _baseline_popularity(val_loader: DataLoader, idx2ent: dict, epochs: int):
    baseline_name = 'BASELINE_POPULAR'
    tb = SummaryWriter(log_dir=f'{el_util.LOG_PATH}/{baseline_name}')

    entity_popularity = _get_entity_popularities(idx2ent)
    calculator_ACC = ACCMetricCalculator()

    for _, y_val in tqdm(val_loader, desc=baseline_name):
        entity_counts = defaultdict(int)
        correct_predictions_by_threshold = {t: defaultdict(int) for t in ACC_THRESHOLDS}
        for entity_indices in y_val:  # go through batch
            pos_ent_index = entity_indices[0]
            known_entity = pos_ent_index.item() != -1

            neg_ent_indices = _remove_padding(entity_indices[1:])
            has_negatives = len(neg_ent_indices) > 0

            entity_counts[(known_entity, has_negatives)] += 1

            if not has_negatives:
                # if no negatives are available, we would either choose the positive or create a new entity -> correct
                for t in correct_predictions_by_threshold:
                    correct_predictions_by_threshold[t][(known_entity, has_negatives)] += 1
                continue

            if not known_entity:
                # unknown entity and negatives available -> prediction always false
                continue

            pos_ent_degree = entity_popularity[pos_ent_index.item()]
            neg_ent_degrees = [entity_popularity[i.item()] for i in neg_ent_indices]
            if pos_ent_degree > max(neg_ent_degrees):
                # correct prediction if positive has higher degree than any negative
                for t in correct_predictions_by_threshold:
                    correct_predictions_by_threshold[t][(known_entity, has_negatives)] += 1
        calculator_ACC.add_predictions((entity_counts, correct_predictions_by_threshold))

    for t in range(epochs):
        calculator_ACC.write_losses(tb, t)
    tb.close()


def _get_entity_popularities(idx2ent: dict) -> dict:
    entity_popularity = {}
    for idx, ent in idx2ent.items():
        ent_uri = dbp_util.name2resource(ent)
        out_degree = sum(len(vals) for vals in dbp_store.get_properties(ent_uri).values())
        in_degree = sum(len(vals) for vals in dbp_store.get_inverse_properties(ent_uri).values())
        entity_popularity[idx] = out_degree + in_degree
    return entity_popularity


def _remove_padding(t: Tensor, padding_value=-1) -> Tensor:
    padding_indices = (t == padding_value).nonzero()
    return t if len(padding_indices) == 0 else t[:padding_indices[0]]

