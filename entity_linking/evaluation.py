from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import normalized_mutual_info_score
from entity_linking.data import Alignment, CandidateAlignment
from entity_linking.matching.util import MatchingScenario
import utils


class PrecisionRecallF1Evaluator:
    def __init__(self, approach_name: str, scenario: MatchingScenario, eval_nil: bool):
        self.approach_name = approach_name
        self.scenario = scenario
        self.eval_nil = eval_nil

    def compute_and_log_metrics(self, prefix: str, prediction: CandidateAlignment, alignment: Alignment, runtime: int):
        prefix += self.scenario.value
        self._compute_and_log_metrics_for_partition(prefix, prediction, alignment, runtime, None)
        if self.eval_nil:
            self._compute_and_log_metrics_for_partition(prefix + '-NIL', prediction, alignment, runtime, True)
            self._compute_and_log_metrics_for_partition(prefix + '-nonNIL', prediction, alignment, runtime, False)

    def _compute_and_log_metrics_for_partition(self, prefix: str, prediction: CandidateAlignment, alignment: Alignment, runtime: int, nil_flag: Optional[bool]):
        utils.get_logger().debug(f'Computing metrics for {prefix}..')
        if self.scenario.is_MM():
            actual_count = alignment.get_mm_match_count(nil_flag)
            pred_count, tp = prediction.get_mm_preds_and_overlap(alignment, nil_flag)
            nmi = None
        else:
            actual_count = alignment.get_me_match_count(nil_flag)
            pred_count, tp = prediction.get_me_preds_and_overlap(alignment, nil_flag)
            clustering = prediction.get_entity_clusterings(alignment, nil_flag)
            nmi = normalized_mutual_info_score(*clustering) if clustering else None
        mention_count = alignment.mention_count(nil_flag)
        entity_count = alignment.entity_count(nil_flag)
        metrics = self._compute_metrics(pred_count, actual_count, tp, runtime, mention_count, entity_count, nmi)
        self._log_metrics(prefix, metrics)

    @classmethod
    def _compute_metrics(cls, pred_count: int, actual_count: int, tp: int, runtime: int, mention_count: int, entity_count: int, nmi: Optional[float]):
        # base metrics
        if pred_count > 0 and actual_count > 0:
            precision = tp / pred_count
            recall = tp / actual_count
            f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0
        else:
            precision = recall = f1 = 0.0
        metrics = {'0_runtime': runtime, '1_predicted': pred_count, '2_actual': actual_count, '3_precision': precision, '4_recall': recall, '5_f1-score': f1, '6_mentions': mention_count, '7_entities': entity_count}
        if nmi is not None:
            metrics['8_nmi'] = nmi
        return metrics

    def _log_metrics(self, prefix: str, metrics: dict, step: int = 0):
        with SummaryWriter(log_dir=f'./logs/ED/{self.approach_name}') as tb:
            for key, val in metrics.items():
                tb.add_scalar(f'{prefix}/{key}', val, step)
