"""Extract the labels of subject entities from word tokens of listings."""

from typing import Tuple, Dict, List, Callable
import torch
from torch.utils.data import Dataset
import numpy as np
import utils
import datetime
from collections import defaultdict
from .preprocess.word_tokenize import WordTokenizerSpecialToken
from .preprocess.pos_label import POSLabel
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification, IntervalStrategy


# APPLY BERT MODEL
MAX_CHUNKS = 100


def extract_subject_entities(page_chunks: Tuple[list, list, list], tokenizer, model) -> Dict[str, dict]:
    subject_entity_dict = defaultdict(lambda: defaultdict(dict))
    page_token_chunks, page_ws_chunks, _ = page_chunks
    for i in range(0, len(page_token_chunks), MAX_CHUNKS):
        _extract_subject_entity_chunks(page_token_chunks[i:i + MAX_CHUNKS], page_ws_chunks[i:i + MAX_CHUNKS], tokenizer, model, subject_entity_dict)
    return {ts: dict(subject_entity_dict[ts]) for ts in subject_entity_dict}  # convert to standard dicts


def _extract_subject_entity_chunks(page_token_chunks: list, page_ws_chunks: list, tokenizer, model, subject_entity_dict):
    inputs = tokenizer(page_token_chunks, is_split_into_words=True, padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = inputs.offset_mapping
    inputs.pop('offset_mapping')
    inputs.to('cuda')

    model.eval()  # make sure the model is not in training mode anymore
    with torch.no_grad():
        outputs = model(**inputs)

    prediction_batches = torch.argmax(outputs.logits.cpu(), dim=2)
    del inputs, outputs

    for word_tokens, word_token_ws, predictions, offsets in zip(page_token_chunks, page_ws_chunks, prediction_batches, offset_mapping):
        topsection_name, section_name, context_end_idx = _extract_context(word_tokens, word_token_ws)
        # collect entity labels
        predictions = np.array(predictions)
        offsets = np.array(offsets)
        word_predictions = predictions[(offsets[:, 0] == 0) & (offsets[:, 1] != 0)]
        # map predictions
        found_entity = False  # only predict one entity per row/entry
        current_entity_tokens = []
        current_entity_label = POSLabel.NONE.value
        for token, token_ws, label in list(zip(word_tokens,  word_token_ws, word_predictions))[context_end_idx+1:]:
            if label == POSLabel.NONE.value or token in WordTokenizerSpecialToken.all_tokens():
                if current_entity_tokens and not found_entity:
                    entity_name = _tokens2name(current_entity_tokens)
                    if _is_valid_entity_name(entity_name):
                        subject_entity_dict[topsection_name][section_name][entity_name] = current_entity_label
                    found_entity = True
                current_entity_tokens = []
                current_entity_label = POSLabel.NONE.value

                if token in WordTokenizerSpecialToken.item_starttokens():
                    found_entity = False  # reset found_entity if entering a new line
            else:
                current_entity_label = current_entity_label or label
                current_entity_tokens.extend([token, token_ws])
        if current_entity_tokens and not found_entity:
            entity_name = _tokens2name(current_entity_tokens)
            if _is_valid_entity_name(entity_name):
                subject_entity_dict[topsection_name][section_name][entity_name] = current_entity_label


def _extract_context(word_tokens: List[str], word_token_ws: List[str]) -> Tuple[str, str, int]:
    ctx_tokens = []
    for i in range(word_tokens.index(WordTokenizerSpecialToken.CONTEXT_END.value)):
        ctx_tokens.extend([word_tokens[i], word_token_ws[i]])
    ctx_separators = [i for i, x in enumerate(ctx_tokens) if x == WordTokenizerSpecialToken.CONTEXT_SEP.value] + [len(ctx_tokens)]
    top_section_ctx = ctx_tokens[ctx_separators[0]+1:ctx_separators[1]]
    section_ctx = ctx_tokens[ctx_separators[1]+1:ctx_separators[2]]
    context_end_idx = word_tokens.index(WordTokenizerSpecialToken.CONTEXT_END.value)
    return _tokens2name(top_section_ctx), _tokens2name(section_ctx), context_end_idx


def _tokens2name(entity_tokens: List[str]) -> str:
    return ''.join(entity_tokens).lstrip().rstrip(' ,(')


def _is_valid_entity_name(entity_name: str) -> bool:
    return len(entity_name) > 2 and not entity_name.isdigit()


# TRAIN SUBJECT ENTITIY TAGGER


def get_tagging_tokenizer_and_model(training_data_retrieval_func: Callable):
    path_to_model = utils._get_cache_path('transformer_for_SE_tagging')
    if not path_to_model.is_dir():
        _train_tagger(training_data_retrieval_func)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = AutoModelForTokenClassification.from_pretrained(path_to_model, output_hidden_states=True)
    model.to('cuda')
    return tokenizer, model


def _train_tagger(training_data_retrieval_func: Callable):
    pretrained_model = utils.get_config('subject_entity.model_se_tagging')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, add_prefix_space=True, additional_special_tokens=list(WordTokenizerSpecialToken.all_tokens()))

    tokens, labels = training_data_retrieval_func()
    train_dataset = _get_dataset(tokens, labels, tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(pretrained_model, num_labels=len(POSLabel))
    model.resize_token_embeddings(len(tokenizer))

    run_id = '{}_{}'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), utils.get_config('logging.filename'))
    training_args = TrainingArguments(
        save_strategy=IntervalStrategy.NO,
        output_dir=f'/tmp',
        logging_dir=f'./logs/transformers/tagging_{run_id}',
        logging_steps=500,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    path_to_model = utils._get_cache_path('transformer_for_SE_tagging')
    model.save_pretrained(path_to_model)
    tokenizer.save_pretrained(path_to_model)


def _get_dataset(tokens: List[List[str]], tags: List[List[str]], tokenizer) -> Dataset:
    train_encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    train_labels = _encode_labels(tags, train_encodings)

    train_encodings.pop('offset_mapping')  # we don't want to pass this to the model
    train_dataset = ListpageDataset(train_encodings, train_labels)
    return train_dataset


def _encode_labels(labels: List[List[str]], encodings) -> List[List[str]]:
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        relevant_label_mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
        truncated_label_length = len(doc_enc_labels[relevant_label_mask])
        if len(doc_labels) < truncated_label_length:
            # uncased tokenizers can be confused by Japanese/Chinese signs leading to an inconsistency between tokens
            # and labels after tokenization. we handle that gracefully by simply filling it up with non-empty labels.
            doc_labels += [POSLabel.NONE.value] * (truncated_label_length - len(doc_labels))
        doc_enc_labels[relevant_label_mask] = doc_labels[:truncated_label_length]
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


class ListpageDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
