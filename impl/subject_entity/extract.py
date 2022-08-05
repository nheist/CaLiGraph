"""Extract the labels of subject entities from word tokens of listings."""

from typing import Tuple, Dict, List, Callable
import torch
from torch.utils.data import Dataset
import numpy as np
import utils
import datetime
from collections import defaultdict
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification, IntervalStrategy
from ..util.transformer import SpecialToken
from .preprocess.pos_label import POSLabel
from .preprocess import sample


# APPLY BERT MODEL
MAX_CHUNKS = 100


def extract_subject_entities(tokenizer, model, chunks: Tuple[list, list, list, list]) -> Dict[int, dict]:
    context_chunks, token_chunks, ws_chunks, _ = chunks
    subject_entity_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for chunk_idx in tqdm(range(0, len(token_chunks), MAX_CHUNKS), desc='Predicting subject entities'):
        chunk_end_idx = chunk_idx + MAX_CHUNKS
        contexts = context_chunks[chunk_idx:chunk_end_idx]
        tokens = token_chunks[chunk_idx:chunk_end_idx]
        ws = ws_chunks[chunk_idx:chunk_end_idx]
        _extract_subject_entity_chunks(contexts, tokens, ws, tokenizer, model, subject_entity_dict)
    return subject_entity_dict


def _extract_subject_entity_chunks(context_chunks: list, token_chunks: list, ws_chunks: list, tokenizer, model, subject_entity_dict: dict):
    inputs = tokenizer(token_chunks, is_split_into_words=True, padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = inputs.offset_mapping
    inputs.pop('offset_mapping')
    inputs.to('cuda')

    model.eval()  # make sure the model is not in training mode anymore
    with torch.no_grad():
        outputs = model(**inputs)

    prediction_batches = torch.argmax(outputs.logits.cpu(), dim=2)
    del inputs, outputs

    for context, word_tokens, word_token_ws, predictions, offsets in zip(context_chunks, token_chunks, ws_chunks, prediction_batches, offset_mapping):
        page_idx = context['page_idx']
        topsection = context['topsection']
        section = context['section']
        # collect entity labels
        predictions = np.array(predictions)
        offsets = np.array(offsets)
        word_predictions = predictions[(offsets[:, 0] == 0) & (offsets[:, 1] != 0)]
        # map predictions
        context_end_idx = word_tokens.index(SpecialToken.CONTEXT_END.value)
        found_entity = False  # only predict one entity per row/entry
        current_entity_tokens = []
        current_entity_label = POSLabel.NONE.value
        for token, token_ws, label in list(zip(word_tokens,  word_token_ws, word_predictions))[context_end_idx+1:]:
            if label == POSLabel.NONE.value or token in SpecialToken.all_tokens():
                if current_entity_tokens and not found_entity:
                    entity_name = _tokens2name(current_entity_tokens)
                    if _is_valid_entity_name(entity_name):
                        subject_entity_dict[page_idx][topsection][section][entity_name] = current_entity_label
                    found_entity = True
                current_entity_tokens = []
                current_entity_label = POSLabel.NONE.value
                if token in SpecialToken.item_starttokens():
                    found_entity = False  # reset found_entity if entering a new line
            else:
                current_entity_label = current_entity_label or label
                current_entity_tokens.extend([token, token_ws])
        if current_entity_tokens and not found_entity:
            entity_name = _tokens2name(current_entity_tokens)
            if _is_valid_entity_name(entity_name):
                subject_entity_dict[page_idx][topsection][section][entity_name] = current_entity_label


def _tokens2name(entity_tokens: List[str]) -> str:
    return ''.join(entity_tokens).lstrip().rstrip(' ,(')


def _is_valid_entity_name(entity_name: str) -> bool:
    return len(entity_name) > 2 and not entity_name.isdigit()


# TRAIN SUBJECT ENTITIY TAGGER


def get_tagging_tokenizer_and_model():
    path_to_model = utils._get_cache_path('transformer_for_mention_detection')
    if not path_to_model.is_dir():
        _train_tagger()
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = AutoModelForTokenClassification.from_pretrained(path_to_model, output_hidden_states=True)
    model.to('cuda')
    return tokenizer, model


def _train_tagger():
    pretrained_model = utils.get_config('subject_entity.model_mention_detection')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, add_prefix_space=True, additional_special_tokens=list(SpecialToken.all_tokens()))
    train_dataset = sample.get_mention_detection_listpage_training_dataset(tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(pretrained_model, num_labels=len(POSLabel))
    model.resize_token_embeddings(len(tokenizer))

    run_id = '{}_{}'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), utils.get_config('logging.filename'))
    training_args = TrainingArguments(
        save_strategy=IntervalStrategy.NO,
        output_dir=f'/tmp',
        logging_dir=f'./logs/transformers/MD_{run_id}',
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
    path_to_model = utils._get_cache_path('transformer_for_mention_detection')
    model.save_pretrained(path_to_model)
    tokenizer.save_pretrained(path_to_model)

