from typing import Tuple, Dict, Optional, Set, List
import random
import utils
from impl.util.rdf import EntityIndex
from impl import subject_entity
from impl.subject_entity import combine
from impl.dbpedia.resource import DbpResource, DbpResourceStore
from impl.subject_entity.preprocess.word_tokenize import WordTokenizer


def get_train_and_val_data(sample: int, items_per_chunk: int) -> Tuple[Dict[DbpResource, tuple], Dict[DbpResource, tuple]]:
    train_pages, val_pages = utils.load_or_create_cache('entity_linking_pages', _load_page_data)
    # draw sample of training data
    sample_fraction = sample / 100  # sample is given as percentage
    train_sample_resources = random.sample(list(train_pages), int(len(train_pages) * sample_fraction))
    train_sample = {res: train_pages[res] for res in train_sample_resources}
    # extract data for matching
    train_data = _create_vector_prediction_data(train_sample, items_per_chunk, False)
    val_data = _create_vector_prediction_data(val_pages, items_per_chunk, True)
    return train_data, val_data


def _load_page_data() -> Tuple[Dict[DbpResource, dict], Dict[DbpResource, dict]]:
    valid_res_indices = set(DbpResourceStore.instance().get_embedding_vectors())
    subject_entity_pages = combine.get_subject_entity_page_content(subject_entity._get_subject_entity_predictions())
    # filter out pages whose main entity has no embedding
    subject_entity_pages = {res: content for res, content in subject_entity_pages.items() if res.idx in valid_res_indices}
    # filter out listings of pages that have no labeled entities at all (because they violate the distant supervision assumption)
    subject_entity_pages = {res: _filter_listings_without_seen_ents(content) for res, content in subject_entity_pages.items()}
    subject_entity_pages = {res: content for res, content in subject_entity_pages.items() if content is not None}  # ignore pages that have no more entities
    # split into train and validation (we use a fixed 5% for validation independent of training size)
    val_sample_resouces = set(random.sample(list(subject_entity_pages), int(len(subject_entity_pages) * .05)))
    train_pages = {res: content for res, content in subject_entity_pages.items() if res not in val_sample_resouces}
    val_pages = {res: subject_entity_pages[res] for res in val_sample_resouces}
    return train_pages, val_pages


def _filter_listings_without_seen_ents(page_content: dict) -> Optional[dict]:
    has_ents = False
    for s in page_content['sections']:
        valid_enums = [enum for enum in s['enums'] if any('subject_entity' in entry and entry['subject_entity']['idx'] != EntityIndex.NEW_ENTITY.value for entry in enum)]
        s['enums'] = valid_enums
        valid_tables = [t for t in s['tables'] if any('subject_entity' in cell and cell['subject_entity']['idx'] != EntityIndex.NEW_ENTITY.value for row in t['data'] for cell in row)]
        s['tables'] = valid_tables
        if valid_enums or valid_tables:
            has_ents = True
    return page_content if has_ents else None


def _create_vector_prediction_data(subject_entity_pages: Dict[DbpResource, dict], items_per_chunk: int, include_new_entities: bool) -> Dict[DbpResource, Tuple[List[List[str]], List[List[str]], List[List[int]]]]:
    entity_labels = _get_subject_entity_labels(subject_entity_pages, include_new_entities)
    return WordTokenizer(max_items_per_chunk=items_per_chunk, max_ents_per_item=1)(subject_entity_pages, entity_labels=entity_labels)


def _get_subject_entity_labels(subject_entity_pages: Dict[DbpResource, dict], include_new_entities: bool) -> Dict[DbpResource, Tuple[Set[int], Set[int]]]:
    valid_res_indices = set(DbpResourceStore.instance().get_embedding_vectors())
    entity_labels = {}
    for res, page_content in subject_entity_pages.items():
        # collect all subject entity labels
        subject_entity_indices = {entry['subject_entity']['idx'] for s in page_content['sections'] for enum in s['enums'] for entry in enum if 'subject_entity' in entry}
        subject_entity_indices.update({cell['subject_entity']['idx'] for s in page_content['sections'] for table in s['tables'] for row in table['data'] for cell in row if 'subject_entity' in cell})
        # get rid of non-entities and entities without RDF2vec embeddings (as we can't use them for training/eval)
        subject_entity_indices = subject_entity_indices.intersection(valid_res_indices)
        if include_new_entities:
            subject_entity_indices.add(EntityIndex.NEW_ENTITY.value)
        entity_labels[res] = (subject_entity_indices, set())
    return entity_labels
