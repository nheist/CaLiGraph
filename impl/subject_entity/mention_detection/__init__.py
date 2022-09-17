from typing import Dict, Tuple, List
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import utils
from impl.util.nlp import EntityTypeLabel
from impl.util.transformer import SpecialToken
from impl.wikipedia import WikiPageStore
from .data import get_listpage_training_dataset, get_page_training_dataset, get_page_data
from .model import LISTPAGE_MODEL, PAGE_MODEL, model_exists, load_tokenizer_and_model, train_tokenizer_and_model


PREDICTION_BATCH_SIZE = 200


def detect_mentions():
    subject_entity_mentions = utils.load_or_create_cache('subject_entity_mentions', _extract_subject_entity_mentions)
    WikiPageStore.instance().set_subject_entity_mentions(subject_entity_mentions)


def _extract_subject_entity_mentions() -> Dict[int, Dict[int, Dict[int, Tuple[str, EntityTypeLabel]]]]:
    if not model_exists(PAGE_MODEL):
        if not model_exists(LISTPAGE_MODEL):
            base_model = utils.get_config('subject_entity.model_mention_detection')
            train_tokenizer_and_model(LISTPAGE_MODEL, base_model)
        lp_tokenizer, lp_model = load_tokenizer_and_model(LISTPAGE_MODEL)
        subject_entity_mentions_from_lp = _extract_mentions_for_model(lp_tokenizer, lp_model)
        WikiPageStore.instance().set_subject_entity_mentions(subject_entity_mentions_from_lp)
        train_tokenizer_and_model(PAGE_MODEL, LISTPAGE_MODEL)
    p_tokenizer, p_model = load_tokenizer_and_model(PAGE_MODEL)
    return _extract_mentions_for_model(p_tokenizer, p_model)


def _extract_mentions_for_model(tokenizer, model) -> Dict[int, Dict[int, Dict[int, Tuple[str, EntityTypeLabel]]]]:
    contexts, tokens, whitespaces = get_page_data()
    model.to('cuda')

    subject_entity_mentions = defaultdict(lambda: defaultdict(dict))
    for batch_idx in tqdm(range(0, len(contexts), PREDICTION_BATCH_SIZE), desc='Predicting subject entities'):
        batch_end_idx = batch_idx + PREDICTION_BATCH_SIZE
        batch_contexts = contexts[batch_idx:batch_end_idx]
        batch_tokens = tokens[batch_idx:batch_end_idx]
        batch_whitespaces = whitespaces[batch_idx:batch_end_idx]
        _extract_mentions_from_batch(batch_contexts, batch_tokens, batch_whitespaces, tokenizer, model, subject_entity_mentions)
    # convert result back to regular dict structure
    subject_entity_mentions = {page_idx: dict(page_data) for page_idx, page_data in subject_entity_mentions.items()}
    return subject_entity_mentions


def _extract_mentions_from_batch(context_batch: List[List[Tuple[int, int, int]]], token_batch: List[List[str]], whitespace_batch: List[List[str]], tokenizer, model, subject_entity_mentions: dict):
    input_batch = tokenizer(token_batch, is_split_into_words=True, padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping_batch = input_batch.offset_mapping
    input_batch.pop('offset_mapping')
    input_batch.to('cuda')

    model.eval()  # make sure the model is not in training mode anymore
    with torch.no_grad():
        outputs = model(**input_batch)
    prediction_batch = torch.argmax(outputs.logits.cpu(), dim=2)
    del input_batch, outputs

    for contexts, tokens, whitespaces, predictions, offsets in zip(context_batch, token_batch, whitespace_batch, prediction_batch, offset_mapping_batch):
        # collect entity labels
        predictions = np.array(predictions)
        offsets = np.array(offsets)
        word_predictions = predictions[(offsets[:, 0] == 0) & (offsets[:, 1] != 0)]
        # locate subject entities on item level
        for item_context, item_tokens, item_whitespaces, item_predictions in _split_predictions_into_items(contexts, tokens, whitespaces, word_predictions):
            special_tokens = SpecialToken.all_tokens()
            se_mask = [t not in special_tokens and p > EntityTypeLabel.NONE.value for t, p in zip(item_tokens, item_predictions)] + [False]
            try:
                # find start and end of subject entity (ignore every potential subject entity after the first one)
                se_start_idx = se_mask.index(True)
                se_end_idx = se_mask.index(False, se_start_idx)  # will never throw an error due to the trailing False
                se_tokens = item_tokens[se_start_idx:se_end_idx]
                se_whitespaces = item_whitespaces[se_start_idx:se_end_idx]
                # add subject entity if it has a valid label
                se_label = _tokens2label(se_tokens, se_whitespaces)
                se_tag = EntityTypeLabel(item_predictions[se_start_idx])
                if _is_valid_entity_label(se_label):
                    page_idx, listing_idx, item_idx = item_context
                    subject_entity_mentions[page_idx][listing_idx][item_idx] = (se_label, se_tag)
            except ValueError:
                continue  # no subject entity found -> skip item


def _split_predictions_into_items(contexts: List[Tuple[int, int, int]], tokens: List[str], whitespaces: List[str], word_predictions: List[int]) -> Tuple[Tuple[int, int, int], List[str], List[str], List[int]]:
    # discard context of items in token sequence
    items_start_idx = tokens.index(SpecialToken.CONTEXT_END.value) + 1
    tokens = tokens[items_start_idx:]
    whitespaces = whitespaces[items_start_idx:]
    word_predictions = word_predictions[items_start_idx:]
    # split by start token
    start_tokens = SpecialToken.item_starttokens()
    item_ranges = [idx for idx, token in enumerate(tokens) if token in start_tokens] + [len(tokens)]
    item_ranges = list(zip(item_ranges, item_ranges[1:]))  # convert to tuples with (item_start, item_end)
    assert len(contexts) == len(item_ranges), 'There should be exactly one item context for an item range.'
    for context, (item_start, item_end) in zip(contexts, item_ranges):
        yield context, tokens[item_start:item_end], whitespaces[item_start:item_end], word_predictions[item_start:item_end]


def _tokens2label(tokens: List[str], whitespaces: List[str]) -> str:
    label = [None] * (len(tokens) + len(whitespaces))
    label[::2] = tokens
    label[1::2] = whitespaces
    return ''.join(label).lstrip().rstrip(' ,(')


def _is_valid_entity_label(label: str) -> bool:
    return len(label) > 2 and not label.isdigit()
