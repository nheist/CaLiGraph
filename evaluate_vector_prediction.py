from typing import Dict, List, Tuple, Set, Iterable
import argparse
import random
import numpy as np
import torch
import utils
from impl import subject_entity
from impl.subject_entity import combine
from impl.subject_entity.preprocess.word_tokenize import WordTokenizer, WordTokenizerSpecialToken, WordTokenizerSpecialLabel
from transformers import Trainer, IntervalStrategy, TrainingArguments, AutoTokenizer, AutoModel
from impl.dbpedia.resource import DbpResource
from entity_linking.model.linking import TransformerForEntityVectorPrediction
from entity_linking.data.linking import prepare_linking_dataset


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
    model = TransformerForEntityVectorPrediction(encoder, EMBEDDING_DIM)
    # load data
    train_pages, val_pages = utils.load_or_create_cache('vector_prediction_training_data', _load_train_and_val_pages)
    # prepare data
    train_data = _prepare_data(train_pages.values(), tokenizer, num_ents)
    val_data = _prepare_data(val_pages.values(), tokenizer, num_ents)
    # run evaluation
    training_args = TrainingArguments(
        seed=SEED,
        save_strategy=IntervalStrategy.NO,
        output_dir=f'./vp_eval/output/{run_id}',
        logging_strategy=IntervalStrategy.STEPS,
        logging_dir=f'./vp_eval/logs/{run_id}',
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
#        compute_metrics=  # TODO
    )
    trainer.train()


def _load_train_and_val_pages() -> Tuple[Dict[DbpResource, tuple], Dict[DbpResource, tuple]]:
    subject_entity_pages = combine.get_subject_entity_page_content(subject_entity._get_subject_entity_predictions())
    # split into train and validation
    all_pages = random.sample(list(subject_entity_pages), int(len(subject_entity_pages) * .1))  # only use 10% of overall data for now
    train_pages = set(random.sample(all_pages, int(len(all_pages) * .9)))  # 90% of pages for train, 10% for val
    val_pages = set(all_pages).difference(train_pages)
    return _create_vector_prediction_data({res: content for res, content in subject_entity_pages.items() if res in train_pages}, False), \
           _create_vector_prediction_data({res: content for res, content in subject_entity_pages.items() if res in val_pages}, True)


def _create_vector_prediction_data(subject_entity_pages: Dict[DbpResource, dict], include_new_entities: bool) -> Dict[DbpResource, Tuple[List[List[str]], List[List[int]]]]:
    entity_labels = _get_subject_entity_labels(subject_entity_pages, include_new_entities)
    return WordTokenizer()(subject_entity_pages, entity_labels=entity_labels)


def _get_subject_entity_labels(subject_entity_pages: Dict[DbpResource, dict], include_new_entities: bool) -> Dict[DbpResource, Tuple[Set[int], Set[int]]]:
    entity_labels = {}
    for res, page_content in subject_entity_pages.items():
        subject_entity_indices = set()
        subject_entity_indices.update({ent['idx'] for s in page_content['sections'] for enum in s['enums'] for entry in enum for ent in entry['entities']})
        subject_entity_indices.update({ent['idx'] for s in page_content['sections'] for table in s['tables'] for row in table['data'] for cell in row for ent in cell['entities']})
        if not include_new_entities:
            subject_entity_indices.discard(WordTokenizerSpecialLabel.NEW_ENTITY.value)
        entity_labels[res] = (subject_entity_indices, set())
    return entity_labels


def _prepare_data(page_data: Iterable[Tuple[List[List[str]], List[List[int]]]], tokenizer, num_ents: int):
    tokens, labels = [], []
    for token_chunks, entity_chunks in page_data:
        tokens.extend(token_chunks)
        labels.extend(entity_chunks)
    return prepare_linking_dataset(tokens, labels, tokenizer, num_ents)


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
