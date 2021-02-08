"""Extract the labels of subject entities from word tokens of listings."""

import torch
import re
import numpy as np
import utils
import datetime
from collections import defaultdict
from .tokenize import TOKEN_ROW, TOKENS_ENTRY, TOKEN_CTX, TOKEN_SEP, ADDITIONAL_SPECIAL_TOKENS, ALL_LABEL_IDS
from transformers import Trainer, TrainingArguments, BertTokenizerFast, BertForTokenClassification


# APPLY BERT MODEL
MAX_BATCHES = 100


def extract_subject_entities(page_batches: list, bert_tokenizer, bert_model) -> dict:
    subject_entity_dict = defaultdict(lambda: defaultdict(dict))

    for i in range(0, len(page_batches), MAX_BATCHES):
        _extract_subject_entity_batches(page_batches[i:i+MAX_BATCHES], bert_tokenizer, bert_model, subject_entity_dict)

    return {ts: dict(subject_entity_dict[ts]) for ts in subject_entity_dict}  # convert to standard dict


def _extract_subject_entity_batches(page_batches: list, bert_tokenizer, bert_model, subject_entity_dict):
    inputs = bert_tokenizer(page_batches, is_split_into_words=True, padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = inputs.offset_mapping
    inputs.pop('offset_mapping')
    inputs.to('cuda')

    with torch.no_grad():
        outputs = bert_model(**inputs).logits.cpu()
    prediction_batches = torch.argmax(outputs, dim=2)
    del inputs, outputs

    for word_tokens, predictions, prediction_offsets in zip(page_batches, prediction_batches, offset_mapping):
        predictions = np.array(predictions)
        prediction_offsets = np.array(prediction_offsets)
        word_predictions = predictions[(prediction_offsets[:, 0] == 0) & (prediction_offsets[:, 1] != 0)]
        topsection_name, section_name = _extract_context(word_tokens)

        found_entity = False  # only predict one entity per row/entry
        current_entity = []
        current_entity_label = None
        for token, label in zip(word_tokens, word_predictions):
            if token in (TOKENS_ENTRY + [TOKEN_ROW]):
                found_entity = False
                continue
            if label == 0:
                if current_entity and not found_entity:
                    entity_name = _entity_tokens2name(current_entity)
                    subject_entity_dict[topsection_name][section_name][entity_name] = current_entity_label
                    found_entity = True
                current_entity = []
                current_entity_label = None
            else:
                current_entity_label = current_entity_label or label
                current_entity.append(token)
        if current_entity and not found_entity:
            entity_name = _entity_tokens2name(current_entity)
            subject_entity_dict[topsection_name][section_name][entity_name] = current_entity_label


def _extract_context(word_tokens: list) -> tuple:
    word_tokens = word_tokens[:word_tokens.index(TOKEN_SEP)]
    context_parts = ' '.join(word_tokens).split(TOKEN_CTX)
    return context_parts[1].strip(), context_parts[2].strip()


def _entity_tokens2name(entity_tokens: list) -> str:
    entity_name = ' '.join(entity_tokens)
    entity_name = re.sub(r'\s*,\s*', ', ', entity_name)
    entity_name = re.sub(r'\s*\.\s*', '. ', entity_name)
    entity_name = re.sub(r'\s*\?\s*', '? ', entity_name)
    return entity_name.strip()


# TRAIN BERT MODEL


BERT_BASE_MODEL = 'bert-base-cased'


def get_bert_tokenizer_and_model(training_data_retrieval_func):
    path_to_model = utils._get_cache_path('bert_for_SE_tagging')
    if not path_to_model.is_dir():
        _train_bert(training_data_retrieval_func)
    tokenizer = BertTokenizerFast.from_pretrained(path_to_model)
    model = BertForTokenClassification.from_pretrained(path_to_model)
    model.to('cuda')
    return tokenizer, model


def _train_bert(training_data_retrieval_func):
    tokenizer = BertTokenizerFast.from_pretrained(BERT_BASE_MODEL)
    tokenizer.add_tokens(ADDITIONAL_SPECIAL_TOKENS)

    tokens, labels = training_data_retrieval_func()
    train_dataset = _get_datasets(tokens, labels, tokenizer)

    model = BertForTokenClassification.from_pretrained(BERT_BASE_MODEL, num_labels=len(ALL_LABEL_IDS))
    model.resize_token_embeddings(len(tokenizer))

    run_id = '{}_{}'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), utils.get_config('logging.filename'))
    training_args = TrainingArguments(
        output_dir=f'./bert/results/{run_id}',
        logging_dir=f'./bert/logs/{run_id}',
        logging_steps=500,
        save_steps=2000,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_steps=0,
        weight_decay=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    path_to_model = utils._get_cache_path('bert_for_SE_tagging')
    model.save_pretrained(path_to_model)
    tokenizer.save_pretrained(path_to_model)


def _get_datasets(tokens, tags, tokenizer) -> tuple:
    train_encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    train_labels = _encode_tags(tags, train_encodings)

    train_encodings.pop('offset_mapping')  # we don't want to pass this to the model
    train_dataset = ListpageDataset(train_encodings, train_labels)
    return train_dataset


def _encode_tags(tags, encodings):
    labels = [[ALL_LABEL_IDS[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        truncated_label_length = len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)])
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels[:truncated_label_length]
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


class ListpageDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
