"""Extract the labels of subject entities from word tokens of listings."""

import torch
import numpy as np
import utils
import datetime
from collections import defaultdict
from .preprocess.word_tokenize import BertSpecialToken
from .preprocess.pos_label import POSLabel
from transformers import Trainer, TrainingArguments, BertTokenizerFast, BertForTokenClassification
from typing import Tuple


# APPLY BERT MODEL
MAX_CHUNKS = 100


def extract_subject_entities(page_chunks: Tuple[list, list], bert_tokenizer, bert_model) -> tuple:
    subject_entity_dict = defaultdict(lambda: defaultdict(dict))
    subject_entity_embeddings_dict = defaultdict(lambda: defaultdict(dict))

    page_token_chunks, page_ws_chunks = page_chunks
    for i in range(0, len(page_token_chunks), MAX_CHUNKS):
        _extract_subject_entity_batches(page_token_chunks[i:i + MAX_CHUNKS], page_ws_chunks[i:i + MAX_CHUNKS], bert_tokenizer, bert_model, subject_entity_dict, subject_entity_embeddings_dict)

    # convert to standard dicts
    return {ts: dict(subject_entity_dict[ts]) for ts in subject_entity_dict},\
           {ts: dict(subject_entity_embeddings_dict[ts]) for ts in subject_entity_embeddings_dict}


def _extract_subject_entity_batches(page_token_batches: list, page_ws_batches: list, bert_tokenizer, bert_model, subject_entity_dict, subject_entity_embeddings_dict):
    inputs = bert_tokenizer(page_token_batches, is_split_into_words=True, padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = inputs.offset_mapping
    inputs.pop('offset_mapping')
    inputs.to('cuda')

    bert_model.eval()  # make sure the model is not in training mode anymore
    with torch.no_grad():
        outputs = bert_model(**inputs)

    prediction_batches = torch.argmax(outputs.logits.cpu(), dim=2)
    hidden_state_batches = outputs.hidden_states[11].cpu()  # use second-to-last layer as token embedding
    del inputs, outputs

    topsection_states = defaultdict(list)
    section_states = defaultdict(lambda: defaultdict(list))
    for word_tokens, word_token_ws, predictions, hidden_states, offsets in zip(page_token_batches, page_ws_batches, prediction_batches, hidden_state_batches, offset_mapping):
        topsection_name, section_name = _extract_context(word_tokens, word_token_ws)
        # collect section states for embeddings
        section_state = hidden_states[0].tolist()
        topsection_states[topsection_name].append(section_state)
        section_states[topsection_name][section_name].append(section_state)

        # collect entity labels and states
        predictions = np.array(predictions)
        offsets = np.array(offsets)
        word_predictions = predictions[(offsets[:, 0] == 0) & (offsets[:, 1] != 0)]
        word_hidden_states = hidden_states[(offsets[:, 0] == 0) & (offsets[:, 1] != 0)]

        found_entity = False  # only predict one entity per row/entry
        current_entity_tokens = []
        current_entity_states = torch.tensor([])
        current_entity_label = POSLabel.NONE.value
        for token, token_ws, label, states in zip(word_tokens,  word_token_ws, word_predictions, word_hidden_states):
            if token in BertSpecialToken.all_tokens() and label != POSLabel.NONE.value:
                # ignore current line, as it is likely an error
                label = POSLabel.NONE.value
                found_entity = True

            if label == POSLabel.NONE.value:
                if current_entity_tokens and not found_entity:
                    entity_name = _tokens2name(current_entity_tokens)
                    if _is_valid_entity_name(entity_name):
                        subject_entity_dict[topsection_name][section_name][entity_name] = current_entity_label
                        subject_entity_embeddings_dict[topsection_name][section_name][entity_name] = current_entity_states.numpy().mean(0)
                    found_entity = True
                current_entity_tokens = []
                current_entity_states = torch.tensor([])
                current_entity_label = POSLabel.NONE.value

                if token in BertSpecialToken.item_starttokens():
                    found_entity = False  # reset found_entity if entering a new line
            else:
                current_entity_label = current_entity_label or label
                current_entity_states = torch.cat((current_entity_states, states.unsqueeze(0)))
                current_entity_tokens.extend([token, token_ws])
        if current_entity_tokens and not found_entity:
            entity_name = _tokens2name(current_entity_tokens)
            if _is_valid_entity_name(entity_name):
                subject_entity_dict[topsection_name][section_name][entity_name] = current_entity_label
                subject_entity_embeddings_dict[topsection_name][section_name][entity_name] = current_entity_states.numpy().mean(0)
    # compute embeddings for sections
    for ts, ts_states in topsection_states.items():
        subject_entity_embeddings_dict[ts]['_embedding'] = np.array(ts_states).mean(0)
    for ts, ts_data in section_states.items():
        for s, s_states in ts_data.items():
            subject_entity_embeddings_dict[ts][s]['_embedding'] = np.array(s_states).mean(0)


def _extract_context(word_tokens: list, word_token_ws: list) -> tuple:
    ctx_tokens = []
    for i in range(word_tokens.index(BertSpecialToken.CONTEXT_END.value)):
        ctx_tokens.extend([word_tokens[i], word_token_ws[i]])
    ctx_separators = [i for i, x in enumerate(ctx_tokens) if x == BertSpecialToken.CONTEXT_SEP.value] + [len(ctx_tokens)]
    top_section_ctx = ctx_tokens[ctx_separators[0]+1:ctx_separators[1]]
    section_ctx = ctx_tokens[ctx_separators[1]+1:ctx_separators[2]]
    return _tokens2name(top_section_ctx), _tokens2name(section_ctx)


def _tokens2name(entity_tokens: list) -> str:
    return ''.join(entity_tokens).lstrip().rstrip(' ,(')


def _is_valid_entity_name(entity_name: str) -> bool:
    return len(entity_name) > 2 and not entity_name.isdigit()


# TRAIN BERT MODEL


BERT_BASE_MODEL = 'bert-base-cased'


def get_bert_tokenizer_and_model(training_data_retrieval_func):
    path_to_model = utils._get_cache_path('bert_for_SE_tagging')
    if not path_to_model.is_dir():
        _train_bert(training_data_retrieval_func)
    tokenizer = BertTokenizerFast.from_pretrained(path_to_model)
    model = BertForTokenClassification.from_pretrained(path_to_model, output_hidden_states=True)
    model.to('cuda')
    return tokenizer, model


def _train_bert(training_data_retrieval_func):
    tokenizer = BertTokenizerFast.from_pretrained(BERT_BASE_MODEL)
    tokenizer.add_tokens(list(BertSpecialToken.all_tokens()))

    tokens, labels = training_data_retrieval_func()
    train_dataset = _get_datasets(tokens, labels, tokenizer)

    model = BertForTokenClassification.from_pretrained(BERT_BASE_MODEL, num_labels=POSLabel.label_count())
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
    train_labels = _encode_labels(tags, train_encodings)

    train_encodings.pop('offset_mapping')  # we don't want to pass this to the model
    train_dataset = ListpageDataset(train_encodings, train_labels)
    return train_dataset


def _encode_labels(labels, encodings):
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
