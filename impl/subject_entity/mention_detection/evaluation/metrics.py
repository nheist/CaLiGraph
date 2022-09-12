from typing import List
from copy import deepcopy
from collections import namedtuple, Counter
from transformers import EvalPrediction
from impl.util.transformer import EntityIndex
from impl.subject_entity.mention_detection.data.chunking import LISTING_TYPE_ENUM, LISTING_TYPE_TABLE


Entity = namedtuple('Entity', ['e_type', 'start_offset', 'end_offset'])


class SETagsEvaluator:
    def __init__(self):
        self.result_schema = {
            'strict': Counter({'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}),
            'exact': Counter({'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}),
            'partial': Counter({'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}),
            'ent_type': Counter({'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}),
        }
        self.results = {}

    def evaluate(self, eval_prediction: EvalPrediction, listing_types: List[str]) -> dict:
        self.results = {
            'overall': deepcopy(self.result_schema),
            LISTING_TYPE_ENUM: deepcopy(self.result_schema),
            LISTING_TYPE_TABLE: deepcopy(self.result_schema)
        }

        mentions = eval_prediction.predictions.argmax(-1)
        labels = eval_prediction.label_ids
        masks = labels != EntityIndex.IGNORE.value
        for mention_ids, true_ids, mask, listing_type in zip(mentions, labels, masks, listing_types):
            # remove invalid preds/labels
            mention_ids = mention_ids[mask]
            true_ids = true_ids[mask]
            # run computation
            self.compute_metrics(self._collect_named_entities(mention_ids), self._collect_named_entities(true_ids), listing_type)

        # compute overall stats for listing types
        for metric in self.results['overall']:
            self.results['overall'][metric] = self.results[LISTING_TYPE_ENUM][metric] + self.results[LISTING_TYPE_TABLE][metric]

        return self._compute_precision_recall_wrapper()

    def compute_metrics(self, pred_named_entities: set, true_named_entities: set, listing_type: str):
        # keep track of entities that overlapped
        true_which_overlapped_with_pred = set()

        for pred in pred_named_entities:
            # Check each of the potential scenarios in turn. For scenario explanation see
            # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/

            # Scenario I: Exact match between true and pred
            if pred in true_named_entities:
                true_which_overlapped_with_pred.add(pred)
                self.results[listing_type]['strict']['correct'] += 1
                self.results[listing_type]['ent_type']['correct'] += 1
                self.results[listing_type]['exact']['correct'] += 1
                self.results[listing_type]['partial']['correct'] += 1

            else:
                # check for overlaps with any of the true entities
                found_overlap = False
                for true in true_named_entities:
                    pred_range = set(range(pred.start_offset, pred.end_offset))
                    true_range = set(range(true.start_offset, true.end_offset))

                    # Scenario IV: Offsets match, but entity type is wrong
                    if true.start_offset == pred.start_offset and true.end_offset == pred.end_offset and true.e_type != pred.e_type:
                        self.results[listing_type]['strict']['incorrect'] += 1
                        self.results[listing_type]['ent_type']['incorrect'] += 1
                        self.results[listing_type]['partial']['correct'] += 1
                        self.results[listing_type]['exact']['correct'] += 1
                        true_which_overlapped_with_pred.add(true)
                        found_overlap = True
                        break

                    # check for an overlap i.e. not exact boundary match, with true entities
                    elif true_range.intersection(pred_range):
                        true_which_overlapped_with_pred.add(true)

                        # Scenario V: There is an overlap (but offsets do not match exactly), and the entity type is the same.
                        if pred.e_type == true.e_type:
                            self.results[listing_type]['strict']['incorrect'] += 1
                            self.results[listing_type]['ent_type']['correct'] += 1
                            self.results[listing_type]['partial']['partial'] += 1
                            self.results[listing_type]['exact']['incorrect'] += 1
                            found_overlap = True
                            break

                        # Scenario VI: Entities overlap, but the entity type is different.
                        else:
                            self.results[listing_type]['strict']['incorrect'] += 1
                            self.results[listing_type]['ent_type']['incorrect'] += 1
                            self.results[listing_type]['partial']['partial'] += 1
                            self.results[listing_type]['exact']['incorrect'] += 1
                            found_overlap = True
                            break

                # Scenario II: Entities are spurious (i.e., over-generated).
                if not found_overlap:
                    self.results[listing_type]['strict']['spurious'] += 1
                    self.results[listing_type]['ent_type']['spurious'] += 1
                    self.results[listing_type]['partial']['spurious'] += 1
                    self.results[listing_type]['exact']['spurious'] += 1

        # Scenario III: Entity was missed entirely.
        missed_entities = len(true_named_entities.difference(true_which_overlapped_with_pred))
        self.results[listing_type]['strict']['missed'] += missed_entities
        self.results[listing_type]['ent_type']['missed'] += missed_entities
        self.results[listing_type]['partial']['missed'] += missed_entities
        self.results[listing_type]['exact']['missed'] += missed_entities

    def _compute_precision_recall_wrapper(self):
        final_metrics = {}
        for lt, stats in self.results.items():
            for k, v in stats.items():
                for metric_key, metric_value in self._compute_precision_recall(lt, k, v).items():
                    final_metrics[metric_key] = metric_value
        return final_metrics

    @classmethod
    def _compute_precision_recall(cls, listing_type, eval_schema, eval_data):
        correct = eval_data['correct']
        incorrect = eval_data['incorrect']
        partial = eval_data['partial']
        missed = eval_data['missed']
        spurious = eval_data['spurious']
        actual = correct + incorrect + partial + spurious  # number of annotations produced by the NER system
        possible = correct + incorrect + partial + missed  # number of annotations in the gold-standard which contribute to the final score

        if eval_schema in ['partial', 'ent_type']:
            precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
            recall = (correct + 0.5 * partial) / possible if possible > 0 else 0
        else:
            precision = correct / actual if actual > 0 else 0
            recall = correct / possible if possible > 0 else 0

        return {f'{listing_type}-COUNT': possible, f'{listing_type}-P-{eval_schema}': precision, f'{listing_type}-R-{eval_schema}': recall}

    @classmethod
    def _collect_named_entities(cls, mention_ids) -> set:
        named_entities = set()
        start_offset = None
        ent_type = None

        for offset, mention_id in enumerate(mention_ids):
            if mention_id == 0:
                if ent_type is not None and start_offset is not None:
                    named_entities.add(Entity(ent_type, start_offset, offset))
                    start_offset = None
                    ent_type = None
            elif ent_type is None:
                ent_type = mention_id
                start_offset = offset
        # catches an entity that goes up until the last token
        if ent_type is not None and start_offset is not None:
            named_entities.add(Entity(ent_type, start_offset, len(mention_ids)))
        return named_entities
