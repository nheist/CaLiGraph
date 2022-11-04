from typing import Set, Optional
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from entity_linking.entity_disambiguation.data import Pair, Alignment
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from impl.util.transformer import EntityIndex
from impl.wikipedia import WikiPageStore
from impl.wikipedia.page_parser import MentionId


class PrecisionRecallF1Evaluator:
    def __init__(self, approach_name: str, scenario: MatchingScenario):
        self.approach_name = approach_name
        self.scenario = scenario
        self.wps = WikiPageStore.instance()

    def compute_and_log_metrics(self, prefix: str, predicted_pairs: Set[Pair], alignment: Alignment, runtime: int):
        prefix += self.scenario.value
        self._compute_and_log_metrics_for_partition(prefix, predicted_pairs, alignment, runtime, None)
        self._compute_and_log_metrics_for_partition(prefix, predicted_pairs, alignment, runtime, True)
        self._compute_and_log_metrics_for_partition(prefix, predicted_pairs, alignment, runtime, False)

    def _compute_and_log_metrics_for_partition(self, prefix: str, predicted_pairs: Set[Pair], alignment: Alignment, runtime: int, nil_partition: Optional[bool]):
        if nil_partition:  # partition for NIL only
            prefix += '-NIL'
            predicted_pairs = {p for p in predicted_pairs if self._is_nil_mention(p.source) or (self.scenario.is_MM() and self._is_nil_mention(p.target))}
        else:  # partition for non-NIL only
            prefix +='-nonNIL'
            predicted_pairs = {p for p in predicted_pairs if not self._is_nil_mention(p.source) or (self.scenario.is_MM() and not self._is_nil_mention(p.target))}
        pred_count = len(predicted_pairs)
        actual_count = alignment.mm_match_count(nil_partition) if self.scenario.is_MM() else alignment.me_match_count(nil_partition)
        tp = len({p for p in predicted_pairs if p in alignment})

        metrics = self._compute_metrics(pred_count, actual_count, tp, runtime)
        self._log_metrics(prefix, metrics)
        self._log_roc_curve(prefix, predicted_pairs, alignment)

    def _is_nil_mention(self, mention_id: MentionId) -> bool:
        if mention_id[1] == EntityIndex.NEW_ENTITY:  # NILK dataset
            return mention_id[2] == EntityIndex.NEW_ENTITY
        else:  # LIST dataset
            return self.wps.get_subject_entity(mention_id).entity_idx == EntityIndex.NEW_ENTITY

    def _compute_metrics(self, pred_count: int, actual_count: int, tp: int, runtime: int):
        # base metrics
        if pred_count > 0 and actual_count > 0:
            precision = tp / pred_count
            recall = tp / actual_count
            f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0
        else:
            precision = recall = f1 = 0.0
        metrics = {'0_runtime': runtime, '1_predicted': pred_count, '2_actual': actual_count, '3_precision': precision, '4_recall': recall, '5_f1-score': f1}
        return metrics

    def _log_metrics(self, prefix: str, metrics: dict, step: int = 0):
        with SummaryWriter(log_dir=f'./logs/ED/{self.approach_name}') as tb:
            for key, val in metrics.items():
                tb.add_scalar(f'{prefix}/{key}', val, step)

    def _log_roc_curve(self, prefix: str, predicted_pairs: Set[Pair], alignment: Alignment, step: int = 0):
        pred = [p[2] for p in predicted_pairs]
        actual = [1 if p in alignment else 0 for p in predicted_pairs]
        if not pred or sum(actual) == 0:
            return
        fpr, tpr, _ = roc_curve(actual, pred)
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        with SummaryWriter(log_dir=f'./logs/ED/{self.approach_name}') as tb:
            tb.add_figure(f'{prefix}/5_roc', plt.gcf(), step)
