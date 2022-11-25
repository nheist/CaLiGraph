from typing import Optional, Dict, Tuple, List, Set, Iterable
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import normalized_mutual_info_score
from impl.subject_entity.entity_disambiguation.data.util import is_nil_mention, is_consistent_with_nil_flag, Pair, Alignment, CandidateAlignment
import utils
from impl.wikipedia import MentionId


class AlignmentComparison:
    def __init__(self, prediction: CandidateAlignment, actual: Alignment, predict_unknowns: bool):
        self.prediction = prediction
        self.actual = actual
        self.predict_unknowns = predict_unknowns
        self.clustering = self._init_clustering(prediction, actual)

    def _init_clustering(self, prediction: CandidateAlignment, actual: Alignment) -> Optional[List[Tuple[Set[MentionId], Optional[int]]]]:
        if prediction.clustering is None:
            return None
        clusters_with_known_entity = [cluster for cluster in prediction.clustering if cluster[1] is not None]
        clusters_with_unknown_entity = []
        if self.predict_unknowns:
            # find optimal mapping of clusters to unknown ents with by treating it as Linear Sum Assignment problem
            mention_clusters = [mentions for mentions, ent in prediction.clustering if ent is None]
            mention_cluster_assignment = self._compute_mention_cluster_assignment(mention_clusters, actual)
            clusters_with_unknown_entity = [(mentions, ent) for mentions, ent in zip(mention_clusters, mention_cluster_assignment)]
        return clusters_with_known_entity + clusters_with_unknown_entity

    def _compute_mention_cluster_assignment(self, mention_clusters: List[Set[MentionId]], alignment: Alignment) -> List[Optional[int]]:
        # compute count of actual linked entities per cluster
        mention_cluster_entity_counts = []
        for mentions in mention_clusters:
            cluster_entities = [alignment.mention_to_entity_mapping[m_id] for m_id in mentions]
            unknown_entity_counts = Counter([ent for ent in cluster_entities if ent not in alignment.known_entities])
            mention_cluster_entity_counts.append(unknown_entity_counts)
        # create cost matrix for every cluster based on entity counts (use negatives as we want to maximize entity hits)
        unknown_entities = [ent for ent in alignment.entity_to_mention_mapping if ent not in alignment.known_entities]
        unknown_entity_indices = {ent: idx for idx, ent in enumerate(unknown_entities)}
        mention_cluster_costs = np.zeros((len(mention_clusters), len(unknown_entities)))
        for cluster_idx, ent_counts in enumerate(mention_cluster_entity_counts):
            for ent_id, cnt in ent_counts.items():
                mention_cluster_costs[cluster_idx, unknown_entity_indices[ent_id]] = -cnt
        # find optimal assignment of entities to clusters
        cluster_entities = [None] * len(mention_clusters)
        for cluster_idx, entity_idx in zip(*linear_sum_assignment(mention_cluster_costs)):
            ent_id = unknown_entities[entity_idx]
            if mention_cluster_entity_counts[cluster_idx][ent_id] == 0:
                # discard assignment of entity to cluster if no mention in the cluster is linked to the entity
                continue
            cluster_entities[cluster_idx] = ent_id
        return cluster_entities

    def get_actual_mention_count(self, nil_flag: Optional[bool]) -> int:
        return self.actual.mention_count(nil_flag)

    def get_actual_entity_count(self, nil_flag: Optional[bool]) -> int:
        return self.actual.entity_count(nil_flag)

    def has_clustering(self) -> bool:
        return self.clustering is not None

    def get_mention_clusters(self, nil_flag: Optional[bool]) -> Tuple[List[int], List[int]]:
        pred, actual = [], []
        for cluster_id, (mentions, _) in enumerate(self.clustering):
            if nil_flag is None or any(nil_flag == is_nil_mention(mention) for mention in mentions):
                for mention in mentions:
                    pred.append(cluster_id)
                    actual.append(self.actual.mention_to_entity_mapping[mention])
        return pred, actual

    def get_cluster_count(self, nil_flag: Optional[bool]) -> int:
        if nil_flag is None:
            return len(self.clustering)
        if nil_flag:
            return sum(1 for _, ent_id in self.clustering if ent_id not in self.actual.known_entities)
        else:
            return sum(1 for _, ent_id in self.clustering if ent_id in self.actual.known_entities)

    def get_me_preds_and_overlap(self, nil_flag: Optional[bool] = None) -> Tuple[int, int]:
        # select best entity per mention
        pairs_by_mention = defaultdict(set)
        for pair, score in self.get_me_candidates(nil_flag):
            pairs_by_mention[pair[0]].add((pair, score))
        candidates = [max(pairs, key=lambda x: x[1])[0] for pairs in pairs_by_mention.values()]
        # collect stats
        predictions, overlap = 0, 0
        for pair in candidates:
            predictions += 1
            if self.actual.has_match(pair):
                overlap += 1
        return predictions, overlap

    def get_me_candidates(self, nil_flag: Optional[bool]) -> Iterable[Tuple[Pair, float]]:
        if self.has_clustering():
            for mention_ids, ent_id in self.clustering:
                for m_id in mention_ids:
                    pair = (m_id, ent_id)
                    if is_consistent_with_nil_flag(pair, nil_flag):
                        yield pair, 1
        else:
            yield from self.prediction.get_me_candidates(True, nil_flag)


class MetricsCalculator:
    def __init__(self, approach_name: str, predict_unknowns: bool):
        self.approach_name = approach_name
        self.predict_unknowns = predict_unknowns

    def compute_and_log_metrics(self, prefix: str, prediction: CandidateAlignment, alignment: Alignment, runtime: int):
        alignment_comparison = AlignmentComparison(prediction, alignment, self.predict_unknowns)
        self._compute_and_log_metrics_for_partition(f'{prefix}-ALL', alignment_comparison, runtime, None)
        self._compute_and_log_metrics_for_partition(f'{prefix}-KNOWN', alignment_comparison, runtime, False)
        self._compute_and_log_metrics_for_partition(f'{prefix}-UNKNOWN', alignment_comparison, runtime, True)

    def _compute_and_log_metrics_for_partition(self, prefix: str, alignment_comparison: AlignmentComparison, runtime: int, nil_flag: Optional[bool]):
        utils.get_logger().debug(f'Computing metrics for {prefix}..')
        metrics = self._compute_metrics_for_partition(alignment_comparison, runtime, nil_flag)
        # ensure order in tensorboard by prefixing metrics with running index
        metrics = {f'{idx}_{metric}': value for idx, (metric, value) in enumerate(metrics.items())}
        self._log_metrics(prefix, metrics)

    def _compute_metrics_for_partition(self, alignment_comparison: AlignmentComparison, runtime: int, nil_flag: Optional[bool]) -> Dict[str, float]:
        actual_mention_count = alignment_comparison.get_actual_mention_count(nil_flag)
        actual_entity_count = alignment_comparison.get_actual_entity_count(nil_flag)
        pred_count, tp = alignment_comparison.get_me_preds_and_overlap(nil_flag)
        metrics = {
            'runtime': runtime,
            'mentions': actual_mention_count,
            'entities': actual_entity_count
        }
        metrics |= self._compute_p_r_f1('me', pred_count, actual_mention_count, tp)
        if alignment_comparison.has_clustering():
            metrics['clusters'] = alignment_comparison.get_cluster_count(nil_flag)
            true_clusters, predicted_clusters = alignment_comparison.get_mention_clusters(nil_flag)
            metrics['NMI'] = normalized_mutual_info_score(true_clusters, predicted_clusters)
        return metrics

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
