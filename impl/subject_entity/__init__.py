import impl.listpage.store as list_store
import utils
from tqdm import tqdm
from . import combine, extract
from .preprocess.word_tokenize import WordTokenizer
from.preprocess.pos_label import map_entities_to_pos_labels
from impl import wikipedia
import torch


def get_page_subject_entities(graph) -> dict:
    """Retrieve the extracted entities per page with context."""
    # TODO: merge retrieval and combine step => add disambiguation of *ALL* entities during retrieval
    return combine.match_entities_with_uris(_get_subject_entity_predictions(graph))


def _get_subject_entity_predictions(graph) -> dict:
    global __SUBJECT_ENTITY_PREDICTIONS__
    if '__SUBJECT_ENTITY_PREDICTIONS__' not in globals():
        __SUBJECT_ENTITY_PREDICTIONS__ = utils.load_or_create_cache('subject_entity_predictions', lambda: _make_subject_entity_predictions(graph))
    return __SUBJECT_ENTITY_PREDICTIONS__


def _make_subject_entity_predictions(graph) -> dict:
    tokenizer, model = extract.get_bert_tokenizer_and_model(lambda: _get_training_data(graph))
    predictions, _ = {p: extract.extract_subject_entities(batches, tokenizer, model) for p, batches in tqdm(_get_page_data().items(), desc='Predicting subject entities')}
    torch.cuda.empty_cache()  # flush GPU cache to free GPU for other purposes
    return predictions


def _get_training_data(graph) -> tuple:
    # retrieve or extract page-wise training data
    initializer = lambda: WordTokenizer()(list_store.get_parsed_listpages(), graph=graph)
    training_data = utils.load_or_create_cache('subject_entity_training_data', initializer)
    # flatten training data into chunks and replace entities with their POS tags
    tokens, ent_labels = [], []
    for token_chunks, entity_chunks in training_data.values():
        tokens.extend(token_chunks)
        ent_labels.extend(entity_chunks)
    pos_labels = map_entities_to_pos_labels(ent_labels)
    return tokens, pos_labels


def _get_page_data() -> dict:
    global __SUBJECT_ENTITY_PAGE_DATA__
    if '__SUBJECT_ENTITY_PAGE_DATA__' not in globals():
        initializer = WordTokenizer()(wikipedia.get_parsed_articles())
        __SUBJECT_ENTITY_PAGE_DATA__ = utils.load_or_create_cache('subject_entity_page_data', initializer)
    return __SUBJECT_ENTITY_PAGE_DATA__
