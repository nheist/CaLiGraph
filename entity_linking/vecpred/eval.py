import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import Tensor
import entity_linking.vp_utils as el_util
from collections import defaultdict


ACC_THRESHOLD_NONE = -1
ACC_THRESHOLDS = [ACC_THRESHOLD_NONE] + list(np.linspace(0, 1, 11))


class ACCMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=el_util.EPS)
        self.cosine_thresholds = ACC_THRESHOLDS

    def forward(self, input: Tensor, target: Tensor):
        entity_counts = defaultdict(int)
        correct_predictions_by_threshold = {t: defaultdict(int) for t in self.cosine_thresholds}
        for predicted_vector, (entity_vectors, entity_mask) in zip(input, target):  # iterate over input batch
            entity_mask = entity_mask[:, 0].bool()  # entity is valid if respective first element is 1, invalid if 0
            is_known_entity = entity_mask[0].item()

            neg_ent_vectors = entity_vectors[1:][entity_mask[1:]]
            has_negatives = len(neg_ent_vectors) > 0

            entity_counts[(is_known_entity, has_negatives)] += 1

            if not is_known_entity and not has_negatives:
                # any prediction would be correct in this case
                for t in self.cosine_thresholds:
                    correct_predictions_by_threshold[t][(is_known_entity, has_negatives)] += 1
                continue

            pos_ent_vector = entity_vectors[0]
            pos_similarity = self.cos_similarity(predicted_vector.unsqueeze(0), pos_ent_vector.unsqueeze(0)).item()

            if not has_negatives:
                # if no negatives, prediction is correct if similarity to positive is greater than threshold
                for t in self.cosine_thresholds:
                    if pos_similarity > t:
                        correct_predictions_by_threshold[t][(is_known_entity, has_negatives)] += 1
                continue

            neg_similarities = self.cos_similarity(predicted_vector.unsqueeze(0), neg_ent_vectors)
            max_neg_similarity = torch.max(neg_similarities).item()
            if not is_known_entity:
                # if no positive, prediction is correct if no negative is within the threshold
                for t in self.cosine_thresholds:
                    if max_neg_similarity <= t:
                        correct_predictions_by_threshold[t][(is_known_entity, has_negatives)] += 1
                continue

            # otherwise, prediction is correct if similarity to positive is greater than to any negative or threshold
            for t in self.cosine_thresholds:
                if pos_similarity > max(max_neg_similarity, t):
                    correct_predictions_by_threshold[t][(is_known_entity, has_negatives)] += 1
        return entity_counts, correct_predictions_by_threshold


class ACCMetricCalculator:
    def __init__(self):
        super().__init__()
        self.entity_counts = defaultdict(int)
        self.correct_predictions_by_threshold = defaultdict(lambda: defaultdict(int))

    def add_predictions(self, prediction_tuple):
        entity_counts, correct_predictions_by_threshold = prediction_tuple
        for status, val in entity_counts.items():
            self.entity_counts[status] += val
        for t, cp in correct_predictions_by_threshold.items():
            for status, val in cp.items():
                self.correct_predictions_by_threshold[t][status] += val

    def write_losses(self, tb: SummaryWriter, epoch: int):
        ents_total = sum(self.entity_counts.values())
        ents_known_negatives = self.entity_counts[(True, True)]
        ents_known_nonegatives = self.entity_counts[(True, False)]
        ents_unknown_negatives = self.entity_counts[(False, True)]
        ents_unknown_nonegatives = self.entity_counts[(False, False)]

        for t, cp in self.correct_predictions_by_threshold.items():
            cp_total = sum(cp.values())
            cp_known_negatives = cp[(True, True)]
            cp_known_nonegatives = cp[(True, False)]
            cp_unknown_negatives = cp[(False, True)]
            cp_unknown_nonegatives = cp[(False, False)]

            acc_total = cp_total / ents_total
            acc_known = (cp_known_negatives + cp_known_nonegatives) / (ents_known_negatives + ents_known_nonegatives + el_util.EPS)
            acc_unknown = (cp_unknown_negatives + cp_unknown_nonegatives) / (ents_unknown_negatives + ents_unknown_nonegatives + el_util.EPS)
            acc_negatives = (cp_known_negatives + cp_unknown_negatives) / (ents_known_negatives + ents_unknown_negatives + el_util.EPS)
            acc_nonegatives = (cp_known_nonegatives + cp_unknown_nonegatives) / (ents_known_nonegatives + ents_unknown_nonegatives + el_util.EPS)
            acc_known_negatives = cp_known_negatives / (ents_known_negatives + el_util.EPS)
            acc_known_nonegatives = cp_known_nonegatives / (ents_known_nonegatives + el_util.EPS)
            acc_unknown_negatives = cp_unknown_negatives / (ents_unknown_negatives + el_util.EPS)

            tb.add_scalar(self._get_loss_name('0_all', t), acc_total, epoch)
            tb.add_scalar(self._get_loss_name('1_known', t), acc_known, epoch)
            tb.add_scalar(self._get_loss_name('2_unknown', t), acc_unknown, epoch)
            tb.add_scalar(self._get_loss_name('3_negatives', t), acc_negatives, epoch)
            tb.add_scalar(self._get_loss_name('4_nonegatives', t), acc_nonegatives, epoch)
            tb.add_scalar(self._get_loss_name('5_known_negatives', t), acc_known_negatives, epoch)
            tb.add_scalar(self._get_loss_name('6_known_nonegatives', t), acc_known_nonegatives, epoch)
            tb.add_scalar(self._get_loss_name('7_unknown_negatives', t), acc_unknown_negatives, epoch)

    def _get_loss_name(self, loss_type: str, threshold: float):
        return f'ACC/{loss_type}' if threshold == ACC_THRESHOLD_NONE else f'ACC-T{threshold:.1f}/{loss_type}'
