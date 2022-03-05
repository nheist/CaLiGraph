import argparse
import utils
from typing import Tuple
import random
import numpy as np
import torch
from impl.subject_entity.preprocess.word_tokenize import BertSpecialToken
from impl.subject_entity.preprocess.pos_label import POSLabel, map_entities_to_pos_labels
from transformers import Trainer, IntervalStrategy, TrainingArguments, BertTokenizerFast, BertForTokenClassification, EvalPrediction
from impl.subject_entity import extract
from collections import namedtuple
from copy import deepcopy


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def run_evaluation(model: str, learning_rate: float, warmup_steps: int, weight_decay: float, predict_pos_tags: bool):
    run_id = f'{model}_lr-{learning_rate}_ws-{warmup_steps}_wd-{weight_decay}_pp-{predict_pos_tags}'
    # prepare tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(model)
    tokenizer.add_tokens(list(BertSpecialToken.all_tokens()))
    num_labels = len(POSLabel) if predict_pos_tags else 3
    model = BertForTokenClassification.from_pretrained(model, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))
    # load data
    data = utils.load_cache('subject_entity_training_data')
    # split into train and validation
    all_pages = set(data)
    train_pages = set(random.sample(all_pages, int(len(all_pages) * .9)))  # 90% of pages for train, 10% for val
    val_pages = all_pages.difference(train_pages)
    # prepare data
    train_data = _prepare_dataset(tokenizer, [data[p] for p in train_pages], predict_pos_tags)
    val_data = _prepare_dataset(tokenizer, [data[p] for p in val_pages], predict_pos_tags)
    # run evaluation
    training_args = TrainingArguments(
        seed=SEED,
        save_strategy=IntervalStrategy.NO,
        output_dir=f'./se_eval/output/{run_id}',  # should not be used
        logging_strategy=IntervalStrategy.STEPS,
        logging_dir=f'./se_eval/logs/{run_id}',
        logging_steps=500,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=3000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=lambda eval_prediction: SETagsEvaluator(eval_prediction).evaluate()
    )
    trainer.train()


def _prepare_dataset(tokenizer, page_data: list, predict_pos_tags: bool) -> Tuple[list, list]:
    """Flatten data into chunks, assign correct labels, and create dataset"""
    tokens, ent_labels = [], []
    for token_chunks, entity_chunks in page_data:
        tokens.extend(token_chunks)
        ent_labels.extend(entity_chunks)
    labels = map_entities_to_pos_labels(ent_labels) if predict_pos_tags else _map_entity_chunks_to_binary_labels(ent_labels)
    return extract._get_datasets(tokens, labels, tokenizer)


def _map_entity_chunks_to_binary_labels(entity_chunks: list) -> list:
    return [_map_entity_chunk_to_binary_labels(chunk) for chunk in entity_chunks]


def _map_entity_chunk_to_binary_labels(entity_chunk: list) -> list:
    labels = []
    for idx, ent in enumerate(entity_chunk):
        if ent is None:
            labels.append(0)
        elif idx == 0 or ent != entity_chunk[idx-1]:
            labels.append(1)
        else:
            labels.append(2)
    return labels


Entity = namedtuple("Entity", "e_type start_offset end_offset")


class SETagsEvaluator:
    def __init__(self, eval_prediction: EvalPrediction):
        self.predictions = eval_prediction.predictions
        self.labels = eval_prediction.label_ids
        print(f'Predictions shape: {self.predictions.shape}')
        print(f'labels shape: {self.labels.shape}')

        metrics_results = {
            'correct': 0,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'possible': 0,
            'actual': 0,
            'precision': 0,
            'recall': 0,
        }

        self.results = {
            'strict': deepcopy(metrics_results),
            'exact': deepcopy(metrics_results),
            'partial': deepcopy(metrics_results),
            'ent_type': deepcopy(metrics_results),
        }

    def evaluate(self) -> dict:
        for pred_logits, true_ids in zip(self.predictions, self.labels):
            # remove unnecessary preds/labels
            mask = true_ids != -100
            true_ids = true_ids[mask]
            pred_logits = pred_logits[mask]
            # turn pred logits into predictions
            pred_ids = torch.argmax(pred_logits, -1)

            if len(true_ids) != len(pred_ids):
                raise ValueError("Predicted and actual entities do not have the same length!")

            # compute results for one message
            tmp_results = self.compute_metrics(self._collect_named_entities(pred_ids), self._collect_named_entities(true_ids))

            # accumulate results
            for eval_schema in self.results:
                for metric in self.results[eval_schema]:
                    self.results[eval_schema][metric] += tmp_results[eval_schema][metric]

        return self._compute_precision_recall_wrapper(self.results)

    @classmethod
    def _collect_named_entities(cls, label_ids):
        named_entities = []
        start_offset = None
        ent_type = None

        for offset, label_id in enumerate(label_ids):
            if label_id == 0:
                if ent_type is not None and start_offset is not None:
                    named_entities.append(Entity(ent_type, start_offset, offset - 1))
                    start_offset = None
                    ent_type = None
            elif ent_type is None:
                ent_type = label_id
                start_offset = offset
            elif ent_type != label_id-1:  # begin of entity is always inside-1
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                # start of a new entity
                ent_type = label_id
                start_offset = offset
        # catches an entity that goes up until the last token
        if ent_type is not None and start_offset is not None:
            named_entities.append(Entity(ent_type, start_offset, len(label_ids) - 1))
        return named_entities

    @classmethod
    def compute_metrics(cls, pred_named_entities, true_named_entities):
        eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0, 'precision': 0, 'recall': 0}
        evaluation = {
            'strict': deepcopy(eval_metrics),
            'ent_type': deepcopy(eval_metrics),
            'partial': deepcopy(eval_metrics),
            'exact': deepcopy(eval_metrics)
        }

        # keep track of entities that overlapped
        true_which_overlapped_with_pred = []

        for pred in pred_named_entities:
            found_overlap = False

            # Check each of the potential scenarios in turn. For scenario explanation see
            # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/

            # Scenario I: Exact match between true and pred
            if pred in true_named_entities:
                true_which_overlapped_with_pred.append(pred)
                evaluation['strict']['correct'] += 1
                evaluation['ent_type']['correct'] += 1
                evaluation['exact']['correct'] += 1
                evaluation['partial']['correct'] += 1

            else:
                # check for overlaps with any of the true entities
                for true in true_named_entities:
                    pred_range = range(pred.start_offset, pred.end_offset)
                    true_range = range(true.start_offset, true.end_offset)

                    # Scenario IV: Offsets match, but entity type is wrong
                    if true.start_offset == pred.start_offset and pred.end_offset == true.end_offset and true.e_type != pred.e_type:
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1
                        evaluation['partial']['correct'] += 1
                        evaluation['exact']['correct'] += 1
                        true_which_overlapped_with_pred.append(true)
                        found_overlap = True
                        break

                    # check for an overlap i.e. not exact boundary match, with true entities
                    elif set(true_range).intersection(set(pred_range)):
                        true_which_overlapped_with_pred.append(true)

                        # Scenario V: There is an overlap (but offsets do not match exactly), and the entity type is the same.
                        # 2.1 overlaps with the same entity type
                        if pred.e_type == true.e_type:
                            evaluation['strict']['incorrect'] += 1
                            evaluation['ent_type']['correct'] += 1
                            evaluation['partial']['partial'] += 1
                            evaluation['exact']['incorrect'] += 1
                            found_overlap = True
                            break

                        # Scenario VI: Entities overlap, but the entity type is different.
                        else:
                            evaluation['strict']['incorrect'] += 1
                            evaluation['ent_type']['incorrect'] += 1
                            evaluation['partial']['partial'] += 1
                            evaluation['exact']['incorrect'] += 1
                            found_overlap = True
                            break

                # Scenario II: Entities are spurious (i.e., over-generated).
                if not found_overlap:
                    evaluation['strict']['spurious'] += 1
                    evaluation['ent_type']['spurious'] += 1
                    evaluation['partial']['spurious'] += 1
                    evaluation['exact']['spurious'] += 1

        # Scenario III: Entity was missed entirely.
        for true in true_named_entities:
            if true in true_which_overlapped_with_pred:
                continue
            else:
                evaluation['strict']['missed'] += 1
                evaluation['ent_type']['missed'] += 1
                evaluation['partial']['missed'] += 1
                evaluation['exact']['missed'] += 1
        return evaluation

    @classmethod
    def _compute_precision_recall_wrapper(cls, results):
        final_metrics = {}
        for k, v in results.items():
            for metric_key, metric_value in cls._compute_precision_recall(k, v):
                final_metrics[metric_key] = metric_value
        return final_metrics

    @classmethod
    def _compute_precision_recall(cls, eval_schema, results):
        correct = results['correct']
        incorrect = results['incorrect']
        partial = results['partial']
        missed = results['missed']
        spurious = results['spurious']
        actual = correct + incorrect + partial + spurious  # number of annotations produced by the NER system
        possible = correct + incorrect + partial + missed  # number of annotations in the gold-standard which contribute to the final score

        if eval_schema in ['partial', 'ent_type']:
            precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
            recall = (correct + 0.5 * partial) / possible if possible > 0 else 0
        else:
            precision = correct / actual if actual > 0 else 0
            recall = correct / possible if possible > 0 else 0

        return {f'P-{eval_schema}': precision, f'R-{eval_schema}': recall}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the evaluation of subject entity tagging.')
    parser.add_argument('model', help='Huggingface Transformer model used for tagging')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='learning rate used during training')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='warmup steps during learning')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='weight decay during learning')
    parser.add_argument('-pp', '--predict_pos_tags', action="store_true", help='Predict actual POS tags instead of binary SE/non-SE label')
    args = parser.parse_args()
    run_evaluation(args.model, args.learning_rate, args.warmup_steps, args.weight_decay, args.predict_pos_tags)
