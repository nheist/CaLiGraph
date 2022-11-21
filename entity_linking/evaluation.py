from typing import Optional, Dict
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import normalized_mutual_info_score
from entity_linking.data import Alignment, CandidateAlignment
import utils


class PrecisionRecallF1Evaluator:
    def __init__(self, approach_name: str):
        self.approach_name = approach_name

    def compute_and_log_metrics(self, prefix: str, prediction: CandidateAlignment, alignment: Alignment, runtime: int):
        self._compute_and_log_metrics_for_partition(f'{prefix}-ALL', prediction, alignment, runtime, None)
        self._compute_and_log_metrics_for_partition(f'{prefix}-KNOWN', prediction, alignment, runtime, False)
        self._compute_and_log_metrics_for_partition(f'{prefix}-UNKNOWN', prediction, alignment, runtime, True)

    def _compute_and_log_metrics_for_partition(self, prefix: str, prediction: CandidateAlignment, alignment: Alignment, runtime: int, nil_flag: Optional[bool]):
        utils.get_logger().debug(f'Computing metrics for {prefix}..')
        metrics = {
            'runtime': runtime,
            'mentions': alignment.mention_count(nil_flag),
            'entities': alignment.entity_count(nil_flag)
        }
        mention_clusters = prediction.get_mention_clusters(alignment, nil_flag)
        if mention_clusters is not None:
            metrics['NMI'] = normalized_mutual_info_score(*mention_clusters)
        if prediction.scenario.is_MM():
            actual_count = alignment.get_mm_match_count(nil_flag)
            pred_count, tp = prediction.get_mm_preds_and_overlap(alignment, nil_flag)
            metrics |= self._compute_p_r_f1('mm', pred_count, actual_count, tp)
        if prediction.scenario.is_ME():
            actual_count = alignment.mention_count(nil_flag)
            pred_count, tp = prediction.get_me_preds_and_overlap(alignment, nil_flag)
            metrics |= self._compute_p_r_f1('me', pred_count, actual_count, tp)
        # ensure order in tensorboard by prefixing metrics with running index
        metrics = {f'{idx}_{metric}': value for idx, (metric, value) in enumerate(metrics.items())}
        self._log_metrics(prefix, metrics)

    @classmethod
    def _compute_p_r_f1(cls, prefix: str, pred_count: int, actual_count: int, tp: int) -> Dict[str, float]:
        if pred_count > 0 and actual_count > 0:
            precision = tp / pred_count
            recall = tp / actual_count
            f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0
        else:
            precision = recall = f1 = 0.0
        metrics = {f'{prefix}-predicted': pred_count, f'{prefix}-actual': actual_count, f'{prefix}-P': precision, f'{prefix}-R': recall, f'{prefix}-F1': f1}
        return metrics

    def _log_metrics(self, prefix: str, metrics: dict, step: int = 0):
        with SummaryWriter(log_dir=f'./logs/ED/{self.approach_name}') as tb:
            for key, val in metrics.items():
                tb.add_scalar(f'{prefix}/{key}', val, step)
