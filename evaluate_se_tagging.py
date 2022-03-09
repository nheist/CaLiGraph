import argparse
import utils
import random
import numpy as np
import torch
from impl.subject_entity.preprocess.word_tokenize import TransformerSpecialToken
from impl.subject_entity.preprocess.pos_label import POSLabel, map_entities_to_pos_labels
from transformers import Trainer, IntervalStrategy, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification, EvalPrediction
from collections import namedtuple
from entity_linking.data.datasets import prepare_mentiondetection_dataset
from entity_linking.model.mention_detection import TransformerForMentionDetectionAndTypePrediction


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def run_evaluation(model: str, epochs: int, batch_size: int, learning_rate: float, warmup_steps: int, weight_decay: float, predict_single_tag: bool):
    run_id = f'{model}_e-{epochs}_lr-{learning_rate}_ws-{warmup_steps}_wd-{weight_decay}_st-{predict_single_tag}'
    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True, additional_special_tokens=list(TransformerSpecialToken.all_tokens()))
    if predict_single_tag:
        model = TransformerForMentionDetectionAndTypePrediction(model, len(tokenizer), len(POSLabel))
    else:
        model = AutoModelForTokenClassification.from_pretrained(model, num_labels=len(POSLabel))
        model.resize_token_embeddings(len(tokenizer))
    # load data
    data = utils.load_cache('subject_entity_training_data')
    # split into train and validation
    all_pages = set(data)
    train_pages = set(random.sample(all_pages, int(len(all_pages) * .9)))  # 90% of pages for train, 10% for val
    val_pages = all_pages.difference(train_pages)
    # prepare data
    train_data = _prepare_data(tokenizer, [data[p] for p in train_pages], predict_single_tag)
    val_data = _prepare_data(tokenizer, [data[p] for p in val_pages], predict_single_tag)
    # run evaluation
    training_args = TrainingArguments(
        seed=SEED,
        save_strategy=IntervalStrategy.NO,
        output_dir=f'./se_eval/output/{run_id}',
        logging_strategy=IntervalStrategy.STEPS,
        logging_dir=f'./se_eval/logs/{run_id}',
        logging_steps=500,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=5000,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=lambda eval_prediction: SETagsEvaluator(eval_prediction, predict_single_tag).evaluate()
    )
    trainer.train()


def _prepare_data(tokenizer, page_data: list, predict_single_tag: bool):
    """Flatten data into chunks, assign correct labels, and create dataset"""
    tokens, labels = [], []
    for token_chunks, entity_chunks in page_data:
        tokens.extend(token_chunks)
        label_chunks = map_entities_to_pos_labels(entity_chunks, predict_single_tag)
        labels.extend(label_chunks)
    return prepare_mentiondetection_dataset(tokens, labels, tokenizer, predict_single_tag)


Entity = namedtuple("Entity", "e_type start_offset end_offset")


class SETagsEvaluator:
    def __init__(self, eval_prediction: EvalPrediction, predict_single_tag: bool):
        if predict_single_tag:
            # with mention logits we only predict whether there is a subject entity in this position (1 or 0)
            # so we multiply with type_id to "convert" it back to the notion where we predict types per position
            mention_logits, type_logits = eval_prediction.predictions
            type_ids = np.expand_dims(type_logits.argmax(-1), -1)
            self.mentions = mention_logits.argmax(-1) * type_ids
            # same for labels
            mention_labels = eval_prediction.label_ids[:, 0, :]
            type_labels = np.expand_dims(eval_prediction.label_ids[:, 1, 0], -1)
            self.labels = mention_labels * type_labels
            self.masks = mention_labels != -100
        else:
            self.mentions = eval_prediction.predictions.argmax(-1)
            self.labels = eval_prediction.label_ids
            self.masks = self.labels != -100

        self.results = {
            'strict': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0},
            'exact': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0},
            'partial': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0},
            'ent_type': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0},
        }

    def evaluate(self) -> dict:
        for mention_ids, true_ids, mask in zip(self.mentions, self.labels, self.masks):
            # remove invalid preds/labels
            mention_ids = mention_ids[mask]
            true_ids = true_ids[mask]

            self.compute_metrics(self._collect_named_entities(mention_ids), self._collect_named_entities(true_ids))

        return self._compute_precision_recall_wrapper()

    @classmethod
    def _collect_named_entities(cls, mention_ids):
        named_entities = []
        start_offset = None
        ent_type = None

        for offset, mention_id in enumerate(mention_ids):
            if mention_id == 0:
                if ent_type is not None and start_offset is not None:
                    named_entities.append(Entity(ent_type, start_offset, offset))
                    start_offset = None
                    ent_type = None
            elif ent_type is None:
                ent_type = mention_id
                start_offset = offset
        # catches an entity that goes up until the last token
        if ent_type is not None and start_offset is not None:
            named_entities.append(Entity(ent_type, start_offset, len(mention_ids)))
        return named_entities

    def compute_metrics(self, pred_named_entities, true_named_entities):
        # keep track of entities that overlapped
        true_which_overlapped_with_pred = []

        for pred in pred_named_entities:
            found_overlap = False

            # Check each of the potential scenarios in turn. For scenario explanation see
            # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/

            # Scenario I: Exact match between true and pred
            if pred in true_named_entities:
                true_which_overlapped_with_pred.append(pred)
                self.results['strict']['correct'] += 1
                self.results['ent_type']['correct'] += 1
                self.results['exact']['correct'] += 1
                self.results['partial']['correct'] += 1

            else:
                # check for overlaps with any of the true entities
                for true in true_named_entities:
                    pred_range = range(pred.start_offset, pred.end_offset)
                    true_range = range(true.start_offset, true.end_offset)

                    # Scenario IV: Offsets match, but entity type is wrong
                    if true.start_offset == pred.start_offset and pred.end_offset == true.end_offset and true.e_type != pred.e_type:
                        self.results['strict']['incorrect'] += 1
                        self.results['ent_type']['incorrect'] += 1
                        self.results['partial']['correct'] += 1
                        self.results['exact']['correct'] += 1
                        true_which_overlapped_with_pred.append(true)
                        found_overlap = True
                        break

                    # check for an overlap i.e. not exact boundary match, with true entities
                    elif set(true_range).intersection(set(pred_range)):
                        true_which_overlapped_with_pred.append(true)

                        # Scenario V: There is an overlap (but offsets do not match exactly), and the entity type is the same.
                        # 2.1 overlaps with the same entity type
                        if pred.e_type == true.e_type:
                            self.results['strict']['incorrect'] += 1
                            self.results['ent_type']['correct'] += 1
                            self.results['partial']['partial'] += 1
                            self.results['exact']['incorrect'] += 1
                            found_overlap = True
                            break

                        # Scenario VI: Entities overlap, but the entity type is different.
                        else:
                            self.results['strict']['incorrect'] += 1
                            self.results['ent_type']['incorrect'] += 1
                            self.results['partial']['partial'] += 1
                            self.results['exact']['incorrect'] += 1
                            found_overlap = True
                            break

                # Scenario II: Entities are spurious (i.e., over-generated).
                if not found_overlap:
                    self.results['strict']['spurious'] += 1
                    self.results['ent_type']['spurious'] += 1
                    self.results['partial']['spurious'] += 1
                    self.results['exact']['spurious'] += 1

        # Scenario III: Entity was missed entirely.
        for true in true_named_entities:
            if true not in true_which_overlapped_with_pred:
                self.results['strict']['missed'] += 1
                self.results['ent_type']['missed'] += 1
                self.results['partial']['missed'] += 1
                self.results['exact']['missed'] += 1

    def _compute_precision_recall_wrapper(self):
        final_metrics = {}
        for k, v in self.results.items():
            for metric_key, metric_value in self._compute_precision_recall(k, v).items():
                final_metrics[metric_key] = metric_value
        return final_metrics

    @classmethod
    def _compute_precision_recall(cls, eval_schema, eval_data):
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

        return {f'P-{eval_schema}': precision, f'R-{eval_schema}': recall}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the evaluation of subject entity tagging.')
    parser.add_argument('model', help='Huggingface Transformer model used for tagging')
    parser.add_argument('-e', '--epochs', type=int, default=3, help='epochs to train')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='batch size used in train/eval')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='learning rate used during training')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='warmup steps during learning')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='weight decay during learning')
    parser.add_argument('-st', '--predict_single_tag', action="store_true", help='Predict only a single POS tag per chunk')
    args = parser.parse_args()
    run_evaluation(args.model, args.epochs, args.batch_size, args.learning_rate, args.warmup_steps, args.weight_decay, args.predict_single_tag)
