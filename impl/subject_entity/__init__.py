from collections import defaultdict
import impl.listpage.store as list_store
import impl.listpage.util as list_util
import utils
from tqdm import tqdm
import multiprocessing as mp
from . import combine, extract, tokenize
from impl import wikipedia


def get_listpage_entities(graph, listpage_uri: str) -> dict:
    """Retrieve the extracted entities of a given list page."""
    global __LISTPAGE_ENTITIES__
    if '__LISTPAGE_ENTITIES__' not in globals():
        __LISTPAGE_ENTITIES__ = defaultdict(dict, _extract_listpage_entities(graph))
    return __LISTPAGE_ENTITIES__[listpage_uri]


def _extract_listpage_entities(graph) -> dict:
    # retrieve names and NE tags of subject entities for list pages
    lp_subject_entites = {p: data for p, data in _get_subject_entity_predictions(graph).items() if list_util.is_listpage(p)}
    # enrich subject entities with dbpedia entity information (URIs)
    lp_subject_entities = combine.match_entities_with_uris(lp_subject_entites)
    # get rid of top-section, section, and NE tag information
    lp_subject_entities = {lp: {e: e_info['name'] for ts in lp_subject_entities[lp] for s in lp_subject_entities[lp][ts] for e, e_info in lp_subject_entities[lp][ts][s].items()} for lp in lp_subject_entities}
    return lp_subject_entities


def _get_subject_entity_predictions(graph) -> dict:
    global __SUBJECT_ENTITY_PREDICTIONS__
    if '__SUBJECT_ENTITY_PREDICTIONS__' not in globals():
        __SUBJECT_ENTITY_PREDICTIONS__ = utils.load_or_create_cache('subject_entity_predictions', lambda: _make_subject_entity_predictions(graph))
    return __SUBJECT_ENTITY_PREDICTIONS__


def _make_subject_entity_predictions(graph) -> dict:
    tokenizer, model = extract.get_bert_tokenizer_and_model(lambda: _get_training_data(graph))
    return {p: extract.extract_subject_entities(batches, tokenizer, model) for p, batches in tqdm(_get_page_data().items(), desc='Predicting subject entities')}


def _get_training_data(graph) -> tuple:
    global __SUBJECT_ENTITY_TRAINING_DATA__
    if '__SUBJECT_ENTITY_TRAINING_DATA__' not in globals():
        __SUBJECT_ENTITY_TRAINING_DATA__ = utils.load_or_create_cache('subject_entity_training_data', lambda: _retrieve_training_data(graph))
    return __SUBJECT_ENTITY_TRAINING_DATA__


def _retrieve_training_data(graph) -> tuple:
    train_tokens, train_labels = [], []
    with mp.Pool(processes=utils.get_config('max_cpus')) as pool:
        ctx = [(lp_uri, lp_data, graph) for lp_uri, lp_data in list_store.get_parsed_listpages().items()]
        for token_lists, label_lists in tqdm(pool.imap_unordered(tokenize.page_to_tokens_and_labels, ctx, chunksize=50), desc='Extracting BERT training data', total=len(ctx)):
            train_tokens.extend(token_lists)
            train_labels.extend(label_lists)
    return train_tokens, train_labels


def _get_page_data() -> dict:
    global __SUBJECT_ENTITY_PAGE_DATA__
    if '__SUBJECT_ENTITY_PAGE_DATA__' not in globals():
        __SUBJECT_ENTITY_PAGE_DATA__ = utils.load_or_create_cache('subject_entity_page_data', _retrieve_page_data)
    return __SUBJECT_ENTITY_PAGE_DATA__


def _retrieve_page_data() -> dict:
    return dict([tokenize.page_to_tokens(page_tuple) for page_tuple in tqdm(wikipedia.get_parsed_articles().items(), desc='Extracting BERT page data')])
