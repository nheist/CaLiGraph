from typing import Dict, Tuple, List
from collections import defaultdict
import utils
from tqdm import tqdm
from . import heuristics, extract, combine
from .preprocess.word_tokenize import WordTokenizer
from .preprocess.pos_label import map_entities_to_pos_labels
from impl import wikipedia
import torch
from impl.dbpedia.resource import DbpListpage


def get_page_subject_entities() -> Dict[int, dict]:
    """Retrieve the extracted entities per page with context."""
    # TODO: discard matched entities without types to filter out invalid ones
    return combine.match_entities_with_uris(_get_subject_entity_predictions())


def _get_subject_entity_predictions() -> Dict[int, dict]:
    return defaultdict(dict, utils.load_or_create_cache('subject_entity_predictions', _make_subject_entity_predictions))


def _make_subject_entity_predictions() -> Dict[int, dict]:
    tokenizer, model = extract.get_tagging_tokenizer_and_model(_get_training_data)
    predictions = {page_idx: extract.extract_subject_entities(chunks, tokenizer, model) for page_idx, chunks in tqdm(_get_page_data().items(), desc='Predicting subject entities')}
    torch.cuda.empty_cache()  # flush GPU cache to free GPU for other purposes
    return predictions


def _get_training_data() -> Tuple[List[List[str]], List[List[str]]]:
    # retrieve or extract page-wise training data
    training_data = utils.load_or_create_cache('subject_entity_training_data', _get_tokenized_list_pages_with_entity_labels)
    # flatten training data into chunks and replace entities with their POS tags
    tokens, labels = [], []
    for token_chunks, _, entity_chunks in training_data.values():
        tokens.extend(token_chunks)
        labels.extend(map_entities_to_pos_labels(entity_chunks))
    return tokens, labels


def _get_tokenized_list_pages_with_entity_labels() -> Dict[int, Tuple[list, list, list]]:
    wikipedia_listpages = [wp for wp in wikipedia.get_wikipedia_pages() if isinstance(wp.resource, DbpListpage)]
    entity_labels = {wp.idx: heuristics.find_subject_entities_for_listpage(wp) for wp in wikipedia_listpages}
    return WordTokenizer()(wikipedia_listpages, entity_labels=entity_labels)


def _get_page_data() -> Dict[int, Tuple[list, list, list]]:
    initializer = lambda: WordTokenizer()(wikipedia.get_wikipedia_pages())
    return utils.load_or_create_cache('subject_entity_page_data', initializer)
