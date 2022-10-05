from typing import Tuple, Set
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import EvalPrediction
from entity_linking.entity_disambiguation.data import Pair


# TODO: more fine-grained evaluation results (by known/unknown; by type)


class PrecisionRecallF1Evaluator:
    def __init__(self, approach_name: str):
        self.approach_name = approach_name

    def compute_and_log_metrics(self, prefix: str, predicted_pairs: Set[Pair], actual_pairs: Set[Pair], runtime: int):
        predicted = len(predicted_pairs)
        actual = len(actual_pairs)
        if predicted > 0 and actual > 0:
            tp = len(predicted_pairs.intersection(actual_pairs))
            precision = tp / predicted
            recall = tp / actual
            f1 = 2 * precision * recall / (precision + recall)
        else:
            precision = recall = f1 = 0.0
        self._log_metrics(prefix, {'runtime': runtime, 'predicted': predicted, 'actual': actual, 'P': precision, 'R': recall, 'F1': f1})

    def evaluate_trainer(self, eval_prediction: EvalPrediction) -> Tuple[float, float, float]:
        pass  # TODO

    def _compute_metrics(self, prediction: np.array, actual: np.array, num_positives: int) -> Tuple[float, float, float]:
        tp = (prediction == 1) & (actual == 1)
        fp = (prediction == 1) & (actual == 0)
        precision = tp / (tp + fp)
        recall = tp / num_positives
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def _log_metrics(self, prefix: str, metrics: dict, step: int = 0):
        with SummaryWriter(log_dir=f'./logs/ED/{self.approach_name}') as tb:
            for key, val in metrics.items():
                tb.add_scalar(f'{prefix}/{key}', val, step)
