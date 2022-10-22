from typing import Set, Tuple, Dict
from collections import defaultdict, Counter
from operator import attrgetter
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from entity_linking.entity_disambiguation.data import Pair
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from impl.util.nlp import EntityTypeLabel
from impl.util.transformer import EntityIndex
from impl.wikipedia import WikiPageStore
from impl.dbpedia.resource import DbpResourceStore


class PrecisionRecallF1Evaluator:
    def __init__(self, approach_name: str, scenario: MatchingScenario):
        self.approach_name = approach_name
        self.scenario = scenario
        self.wps = WikiPageStore.instance()
        self.dbr = DbpResourceStore.instance()

    def compute_and_log_metrics(self, prefix: str, predicted_pairs: Set[Pair], actual_pairs: Set[Pair], runtime: int):
        # overall
        self._compute_and_log_metrics_for_partition(prefix, predicted_pairs, actual_pairs, runtime)
        # listing type
        self._compute_and_log_metrics_for_partition(f'{prefix}-LT=', predicted_pairs, actual_pairs, runtime, self._get_listing_type)
        # entity status (known vs. unknown)
        self._compute_and_log_metrics_for_partition(f'{prefix}-ES=', predicted_pairs, actual_pairs, runtime, self._get_entity_status)

    def _compute_and_log_metrics_for_partition(self, prefix: str, predicted_pairs: Set[Pair], actual_pairs: Set[Pair], runtime: int, partition_func = None):
        if partition_func:
            for partition_key, (predicted_pair_partition, actual_pair_partition) in self._make_partitions(predicted_pairs, actual_pairs, partition_func).items():
                partition_prefix = prefix + partition_key
                metrics = self._compute_metrics(predicted_pair_partition, actual_pair_partition, runtime)
                self._log_metrics(partition_prefix, metrics)
        else:
            metrics = self._compute_metrics(predicted_pairs, actual_pairs, runtime)
            self._log_metrics(prefix, metrics)
            self._log_roc_curve(prefix, predicted_pairs, actual_pairs)

    def _make_partitions(self, predicted_pairs: Set[Pair], actual_pairs: Set[Pair], partition_func) -> Dict[str, Tuple[Set[Pair], Set[Pair]]]:
        partitioned_predicted_pairs = self._apply_partition_func(partition_func, predicted_pairs)
        partitioned_actual_pairs = self._apply_partition_func(partition_func, actual_pairs)
        partitioning = {}
        for partition_key in set(partitioned_predicted_pairs) | set(partitioned_actual_pairs):
            partitioning[partition_key] = (partitioned_predicted_pairs[partition_key], partitioned_actual_pairs[partition_key])
        return partitioning

    def _apply_partition_func(self, partition_func, pairs: Set[Pair]) -> Dict[str, Set[Pair]]:
        partitioning = defaultdict(set)
        for pair in pairs:
            partition_key = partition_func(pair.source)
            if self.scenario == MatchingScenario.MENTION_MENTION and partition_key != partition_func(pair.target):
                partition_key = 'Mixed'
            partitioning[partition_key].add(pair)
        return partitioning

    def _get_listing_type(self, item_id: Tuple[int, int, int]) -> str:
        return self.wps.get_page(item_id[0]).listings[item_id[1]].get_type()

    def _get_entity_status(self, item_id: Tuple[int, int, int]) -> str:
        return 'Unknown' if self.wps.get_subject_entity(item_id).entity_idx == EntityIndex.NEW_ENTITY.value else 'Known'

    def _compute_metrics(self, predicted_pairs: Set[Pair], actual_pairs: Set[Pair], runtime: int):
        # base metrics
        predicted = len(predicted_pairs)
        actual = len(actual_pairs)
        if predicted > 0 and actual > 0:
            tp = len(predicted_pairs.intersection(actual_pairs))
            precision = tp / predicted
            recall = tp / actual
            f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0
        else:
            precision = recall = f1 = 0.0
        metrics = {'0_runtime': runtime, '1_predicted': predicted, '2_actual': actual, '3_precision': precision, '4_recall': recall, '5_f1-score': f1}
#        # distributions over types
#        predicted_with_types = {pair: self._get_types_for_pair(pair) for pair in predicted_pairs}
#        actual_with_types = {pair: self._get_types_for_pair(pair) for pair in actual_pairs}
#        # true type distribution
#        predicted_by_type = self._get_pairs_by_matching_type(predicted_with_types)
#        actual_by_type = self._get_pairs_by_matching_type(actual_with_types)
#        for t in set(predicted_by_type) | set(actual_by_type):
#            t_actual = len(actual_by_type[t])
#            t_predicted = len(predicted_by_type[t])
#            t_common = len(actual_by_type[t].intersection(predicted_by_type[t]))
#            metrics |= {
#                f'6_{t.name}-1_predicted': t_predicted,
#                f'6_{t.name}-2_actual': t_actual,
#                f'6_{t.name}-3_precision': t_common / t_predicted if t_predicted else 0,
#                f'6_{t.name}-4_recall': t_common / t_actual if t_actual else 0,
#            }
#        # predicted cross-type distribution
#        cross_type_counts = self._compute_cross_type_counts(predicted_with_types)
#        metrics |= {f'7_{type_a.name}-{type_b.name}': cnt for (type_a, type_b), cnt in cross_type_counts.items()}
        return metrics

    def _get_types_for_pair(self, pair: Pair) -> Tuple[EntityTypeLabel, EntityTypeLabel]:
        type_a = self.wps.get_subject_entity(pair[0]).entity_type
        type_b = self.wps.get_subject_entity(pair[1]).entity_type if self.scenario == MatchingScenario.MENTION_MENTION else self.dbr.get_type_label(pair[1])
        return type_a, type_b

    @classmethod
    def _get_pairs_by_matching_type(cls, pairs_with_types: Dict[Pair, Tuple[EntityTypeLabel, EntityTypeLabel]]):
        pairs_by_type = defaultdict(set)
        for pair, (type_a, type_b) in pairs_with_types.items():
            if type_a == type_b:
                pairs_by_type[type_a].add(pair)
        return pairs_by_type

    def _compute_cross_type_counts(self, pairs_with_types: Dict[Pair, Tuple[EntityTypeLabel, EntityTypeLabel]]) -> Dict[Tuple[EntityTypeLabel, EntityTypeLabel], int]:
        cross_type_counts = Counter()
        for type_a, type_b in pairs_with_types.values():
            if type_a == type_b:
                continue
            if self.scenario == MatchingScenario.MENTION_MENTION:
                type_a, type_b = sorted([type_a, type_b], key=attrgetter('name'))
            cross_type_counts[(type_a, type_b)] += 1
        return cross_type_counts

    def _log_metrics(self, prefix: str, metrics: dict, step: int = 0):
        with SummaryWriter(log_dir=f'./logs/ED/{self.approach_name}') as tb:
            for key, val in metrics.items():
                tb.add_scalar(f'{prefix}/{key}', val, step)

    def _log_roc_curve(self, prefix: str, predicted_pairs: Set[Pair], actual_pairs: Set[Pair], step: int = 0):
        pred = [p[2] for p in predicted_pairs]
        actual = [1 if p in actual_pairs else 0 for p in predicted_pairs]
        fpr, tpr, _ = roc_curve(actual, pred)
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        with SummaryWriter(log_dir=f'./logs/ED/{self.approach_name}') as tb:
            tb.add_figure(f'{prefix}/5_roc', plt.gcf(), step)
