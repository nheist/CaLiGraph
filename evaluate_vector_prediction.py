import os
os.environ['DISABLE_SPACY_CACHE'] = '1'

from typing import Dict, List, Tuple, Set, Iterable
from collections import Counter, defaultdict
import statistics
import argparse
import random
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import utils
from impl.util.rdf import EntityIndex
from impl import subject_entity
from impl.subject_entity import combine
from impl.subject_entity.preprocess.word_tokenize import WordTokenizer, WordTokenizerSpecialToken
from transformers import Trainer, IntervalStrategy, TrainingArguments, AutoTokenizer, AutoModel, EvalPrediction
from impl.dbpedia.resource import DbpResource, DbpResourceStore
from entity_linking.preprocessing.embeddings import EntityIndexToEmbeddingMapper
from entity_linking.model.multi_entity_prediction import TransformerForMultiEntityPrediction
from entity_linking.data.multi_entity_prediction import prepare_dataset


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EMBEDDING_DIM = 200


def run_evaluation(model_name: str, epochs: int, batch_size: int, learning_rate: float, warmup_steps: int, weight_decay: float, num_ents: int):
    run_id = f'{model_name}_e-{epochs}_lr-{learning_rate}_ws-{warmup_steps}_wd-{weight_decay}_ne-{num_ents}'
    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, additional_special_tokens=list(WordTokenizerSpecialToken.all_tokens()))
    encoder = AutoModel.from_pretrained(model_name)
    encoder.resize_token_embeddings(len(tokenizer))
    ent_idx2emb = EntityIndexToEmbeddingMapper(EMBEDDING_DIM)
    model = TransformerForMultiEntityPrediction(encoder, ent_idx2emb, EMBEDDING_DIM)
    # load data
    train_data, val_data = utils.load_or_create_cache('vector_prediction_training_data', lambda: _load_train_and_val_datasets(tokenizer, num_ents))
    # run evaluation
    vp_evaluator = VectorPredictionEvaluator(ent_idx2emb, batch_size)
    training_args = TrainingArguments(
        seed=SEED,
        save_strategy=IntervalStrategy.NO,
        output_dir=f'./vp_eval/output/{run_id}',
        logging_strategy=IntervalStrategy.STEPS,
        logging_dir=f'./vp_eval/logs/{run_id}',
        logging_steps=10,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=500,
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
        compute_metrics=vp_evaluator.evaluate
    )
    trainer.train()


def _load_train_and_val_datasets(tokenizer, num_ents: int) -> Tuple[Dict[DbpResource, tuple], Dict[DbpResource, tuple]]:
    subject_entity_pages = combine.get_subject_entity_page_content(subject_entity._get_subject_entity_predictions())
    # split into train and validation
    all_pages = random.sample(list(subject_entity_pages), int(len(subject_entity_pages) * .1))  # only use 10% of overall data for now

    train_pages = set(random.sample(all_pages, int(len(all_pages) * .9)))  # 90% of pages for train, 10% for val
    train_data = _create_vector_prediction_data({res: content for res, content in subject_entity_pages.items() if res in train_pages}, False)
    train_data = _prepare_data(train_data.values(), tokenizer, num_ents)

    val_pages = set(all_pages).difference(train_pages)
    val_data = _create_vector_prediction_data({res: content for res, content in subject_entity_pages.items() if res in val_pages}, True)
    val_data = _prepare_data(val_data.values(), tokenizer, num_ents)
    return train_data, val_data


def _create_vector_prediction_data(subject_entity_pages: Dict[DbpResource, dict], include_new_entities: bool) -> Dict[DbpResource, Tuple[List[List[str]], List[List[str]], List[List[int]]]]:
    entity_labels = _get_subject_entity_labels(subject_entity_pages, include_new_entities)
    return WordTokenizer()(subject_entity_pages, entity_labels=entity_labels)


def _get_subject_entity_labels(subject_entity_pages: Dict[DbpResource, dict], include_new_entities: bool) -> Dict[DbpResource, Tuple[Set[int], Set[int]]]:
    valid_entity_indices = set(DbpResourceStore.instance().get_embedding_vectors())
    entity_labels = {}
    for res, page_content in subject_entity_pages.items():
        # collect all subject entity labels
        subject_entity_indices = set()
        subject_entity_indices.update({ent['idx'] for s in page_content['sections'] for enum in s['enums'] for entry in enum for ent in entry['entities']})
        subject_entity_indices.update({ent['idx'] for s in page_content['sections'] for table in s['tables'] for row in table['data'] for cell in row for ent in cell['entities']})
        # get rid of non-entities and entities without RDF2vec embeddings (as we can't use them for training/eval)
        subject_entity_indices = subject_entity_indices.intersection(valid_entity_indices)
        if include_new_entities:
            subject_entity_indices.add(EntityIndex.NEW_ENTITY.value)
        entity_labels[res] = (subject_entity_indices, set())
    return entity_labels


def _prepare_data(page_data: Iterable[Tuple[List[List[str]], List[List[int]]]], tokenizer, num_ents: int):
    tokens, labels = [], []
    for token_chunks, _, entity_chunks in page_data:
        tokens.extend(token_chunks)
        labels.extend(entity_chunks)
    return prepare_dataset(tokens, labels, tokenizer, num_ents)


class VectorPredictionEvaluator:
    def __init__(self, ent_idx2emb: EntityIndexToEmbeddingMapper, batch_size: int):
        self.ent_idx2emb = ent_idx2emb
        self.batch_size = batch_size
        self.thresholds = [.1, .3, .5, .7]

    def evaluate(self, eval_prediction: EvalPrediction):
        results = defaultdict(list)
        # gather results batch-wise and then average over them
        labels = torch.from_numpy(eval_prediction.label_ids)  # (batches*bs, num_ents, 2)
        entity_vectors = torch.from_numpy(eval_prediction.predictions)  # (batches*bs, num_ents, ent_dim)
        for label_batch, pred_batch in zip(*[torch.split(t, self.batch_size) for t in [labels, entity_vectors]]):
            entity_labels, entity_status = label_batch[:, 0].reshape(-1), label_batch[:, 1].reshape(-1)
            # compute masks for existing (idx == 0) and new (idx == -1) entities
            known_entity_mask = entity_labels.eq(0)  # (bs*num_ents)
            known_entity_targets = torch.arange(len(known_entity_mask))[known_entity_mask]  # (bs*num_ents)
            unknown_entity_mask = entity_labels.eq(-1)  # (bs*num_ents)
            # retrieve embedding vectors for entity indices
            label_entity_vectors = self.ent_idx2emb(entity_labels)  # (bs*num_ents, ent_dim)
            # compute cosine similarity between predictions and labels
            entity_vectors = pred_batch.view(-1, pred_batch.shape[-1])  # (bs*num_ents, ent_dim)
            entity_similarities = F.normalize(entity_vectors) @ F.normalize(label_entity_vectors).T  # (bs*num_ents, bs*num_ents)
            # compute metrics
            known_entity_count = known_entity_mask.sum().item()
            known_entity_correct_predictions = self._get_correct_predictions_for_known_entities(entity_similarities, known_entity_mask, known_entity_targets)
            unknown_entity_count = unknown_entity_mask.sum().item()
            unknown_entity_correct_predictions = self._get_correct_predictions_for_unknown_entities(entity_similarities, unknown_entity_mask)
            all_entity_count = known_entity_count + unknown_entity_count
            all_entity_correct_predictions = known_entity_correct_predictions + unknown_entity_correct_predictions
            for t in self.thresholds:
                results[f'T-{t}_all'].append(all_entity_correct_predictions[t] / all_entity_count)
                results[f'T-{t}_known'].append(known_entity_correct_predictions[t] / known_entity_count)
                results[f'T-{t}_unknown'].append(unknown_entity_correct_predictions[t] / unknown_entity_count)
        return {label: statistics.fmean(vals) for label, vals in results.items()}  # average over batch results

    def _get_correct_predictions_for_known_entities(self, entity_similarities: Tensor, mask: Tensor, targets: Tensor) -> Counter[float, int]:
        # a prediction for a known entity is correct if its own vector is the most similar and similarity >= threshold
        result = Counter()
        for t in self.thresholds:
            known_entity_similarities = entity_similarities[mask]
            target_most_similar = known_entity_similarities.max(dim=-1).keys.eq(targets)
            target_greater_threshold = known_entity_similarities[torch.arange(len(known_entity_similarities)), targets].ge(t)
            result[t] = target_most_similar.logical_and(target_greater_threshold).sum().item()
        return result

    def _get_correct_predictions_for_unknown_entities(self, entity_similarities: Tensor, mask: Tensor) -> Counter[float, int]:
        # a prediction for a unknown entity is correct if there is no known entity more similar than the threshold
        result = {t: entity_similarities[mask].max(dim=-1).values.lt(t).sum().item() for t in self.thresholds}
        return Counter(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the evaluation of vector prediction.')
    parser.add_argument('model_name', help='Huggingface Transformer model used for prediction')
    parser.add_argument('-e', '--epochs', type=int, default=3, help='epochs to train')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='batch size used in train/eval')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='learning rate used during training')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='warmup steps during learning')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='weight decay during learning')
    parser.add_argument('-ne', '--num_ents', type=float, default=128, help='number of candidate entities per sentence')
    args = parser.parse_args()
    run_evaluation(args.model_name, args.epochs, args.batch_size, args.learning_rate, args.warmup_steps, args.weight_decay, args.num_ents)