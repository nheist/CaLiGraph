from typing import Dict, Tuple, List
import utils
from tqdm import tqdm
from . import combine, extract
from .preprocess.word_tokenize import WordTokenizer
from .preprocess.pos_label import map_entities_to_pos_labels
from impl.listpage.subject_entity import find_subject_entities_for_listpage
from impl import wikipedia
import torch
from impl.dbpedia.resource import DbpResource, DbpResourceStore


def get_page_subject_entities(graph) -> Dict[DbpResource, dict]:
    """Retrieve the extracted entities per page with context."""
    # TODO: merge retrieval and combine step => add disambiguation of *ALL* entities during retrieval
    return combine.match_entities_with_uris(_get_subject_entity_predictions(graph))


def _get_subject_entity_predictions(graph) -> Dict[DbpResource, dict]:
    initializer = lambda: _make_subject_entity_predictions(graph)
    return utils.load_or_create_cache('subject_entity_predictions', initializer)


def _make_subject_entity_predictions(graph) -> Dict[DbpResource, dict]:
    tokenizer, model = extract.get_tagging_tokenizer_and_model(lambda: _get_training_data(graph))
    predictions = {r: extract.extract_subject_entities(chunks, tokenizer, model)[0] for r, chunks in tqdm(_get_page_data().items(), desc='Predicting subject entities')}
    torch.cuda.empty_cache()  # flush GPU cache to free GPU for other purposes
    return predictions


def _get_training_data(graph) -> Tuple[List[List[str]], List[List[str]]]:
    # retrieve or extract page-wise training data
    initializer = lambda: _get_tokenized_list_pages_with_entity_labels(graph)
    training_data = utils.load_or_create_cache('subject_entity_training_data', initializer)
    # flatten training data into chunks and replace entities with their POS tags
    tokens, labels = [], []
    for token_chunks, entity_chunks in training_data.values():
        tokens.extend(token_chunks)
        labels.extend(map_entities_to_pos_labels(entity_chunks))
    return tokens, labels


def _get_tokenized_list_pages_with_entity_labels(graph) -> Dict[DbpResource, Tuple[list, list]]:
    dbr = DbpResourceStore.instance()
    listpages = {dbr.resolve_redirect(lp) for lp in dbr.get_listpages()}
    lps_with_content = {res: content for res, content in wikipedia.get_parsed_articles().items() if res in listpages}
    entity_labels = {lp: find_subject_entities_for_listpage(lp, content, graph) for lp, content in lps_with_content.items()}
    return WordTokenizer()(lps_with_content, entity_labels=entity_labels)


def _get_page_data() -> Dict[DbpResource, Tuple[list, list]]:
    initializer = lambda: WordTokenizer()(wikipedia.get_parsed_articles())
    return utils.load_or_create_cache('subject_entity_page_data', initializer)
