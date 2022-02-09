import impl.listpage.store as list_store
import impl.listpage.nlp as list_nlp
import utils
from tqdm import tqdm
import multiprocessing as mp
from . import combine, extract, tokenize
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
    global __SUBJECT_ENTITY_TRAINING_DATA__
    if '__SUBJECT_ENTITY_TRAINING_DATA__' not in globals():
        __SUBJECT_ENTITY_TRAINING_DATA__ = utils.load_or_create_cache('subject_entity_training_data', lambda: _retrieve_training_data(graph))
    return __SUBJECT_ENTITY_TRAINING_DATA__


def _retrieve_training_data(graph) -> tuple:
    list_nlp.parse('')  # make sure that parser has been trained and initialized before going into multiprocessing

    train_tokens, train_labels = [], []
    with mp.Pool(processes=round(utils.get_config('max_cpus') / 4)) as pool:
        ctx = [(lp_uri, lp_data, graph) for lp_uri, lp_data in list_store.get_parsed_listpages().items()]
        for token_lists, label_lists in tqdm(pool.imap_unordered(tokenize.page_to_tokens_and_labels, ctx, chunksize=200), desc='Extracting BERT training data', total=len(ctx)):
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
