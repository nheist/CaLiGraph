from typing import Tuple, Dict, Set, List
import random
import utils
from impl.util.transformer import EntityIndex
from impl.wikipedia import WikipediaPage
from impl import subject_entity
from impl.subject_entity import combine
from impl.subject_entity.preprocess import sample
from impl.dbpedia.resource import DbpResourceStore
from impl.subject_entity.preprocess.word_tokenize import WordTokenizer, WordTokenizedPage


def get_md_train_and_val_data() -> Tuple[List[WordTokenizedPage], List[WordTokenizedPage]]:
    # TODO: add option to use actual page data for specialization => use only "high-quality" page data for evaluation
    tokenized_list_pages = sample._get_tokenized_list_pages_with_entity_labels()
    # split into train and validation (we use a fixed 20% for validation)
    sample_size = int(len(tokenized_list_pages) * .2)
    val_sample_indices = set(random.sample([wp.idx for wp in tokenized_list_pages], sample_size))
    train_pages = [tlp for tlp in tokenized_list_pages if tlp.idx not in val_sample_indices]
    val_pages = [tlp for tlp in tokenized_list_pages if tlp.idx in val_sample_indices]
    return train_pages, val_pages


def get_mem_train_and_val_data() -> Tuple[List[WordTokenizedPage], List[WordTokenizedPage]]:
    train_pages, val_pages = utils.load_or_create_cache('entity_linking_pages', _load_page_data)
    # extract data for matching
    train_data = _create_mention_entity_matching_data(train_pages, False)
    val_data = _create_mention_entity_matching_data(val_pages, True)
    return train_data, val_data


def _load_page_data() -> Tuple[List[WikipediaPage], List[WikipediaPage]]:
    valid_res_indices = set(DbpResourceStore.instance().get_embedding_vectors())
    wiki_pages = combine.get_subject_entity_page_content(subject_entity._get_subject_entity_predictions())
    # filter out pages whose main entity has no embedding
    wiki_pages = [wp for wp in wiki_pages if wp.idx in valid_res_indices]
    # filter out listings that do not have any entities that we know (as they violate the distant supervision assumption)
    for wp in wiki_pages:
        wp.discard_listings_without_seen_entities()
    # filter out listings of pages that have no labeled subject entities at all
    wiki_pages = [wp for wp in wiki_pages if wp.get_subject_entity_indices()]
    # split into train and validation (we use a fixed 5% for validation independent of training size)
    val_sample_indices = set(random.sample([wp.idx for wp in wiki_pages], int(len(wiki_pages) * .05)))
    train_pages = [wp for wp in wiki_pages if wp.idx not in val_sample_indices]
    val_pages = [wp for wp in wiki_pages if wp.idx in val_sample_indices]
    return train_pages, val_pages


def _create_mention_entity_matching_data(wiki_pages: List[WikipediaPage], include_new_entities: bool) -> List[WordTokenizedPage]:
    entity_labels = _get_subject_entity_labels(wiki_pages, include_new_entities)
    return WordTokenizer(max_ents_per_item=1)(wiki_pages, entity_labels=entity_labels)


def _get_subject_entity_labels(wiki_pages: List[WikipediaPage], include_new_entities: bool) -> Dict[int, Tuple[Set[int], Set[int]]]:
    valid_res_indices = set(DbpResourceStore.instance().get_embedding_vectors())
    entity_labels = {}
    for wp in wiki_pages:
        subject_entity_indices = wp.get_subject_entity_indices()
        # get rid of non-entities and entities without RDF2vec embeddings (as we can't use them for training/eval)
        subject_entity_indices.intersection_update(valid_res_indices)
        if include_new_entities:
            subject_entity_indices.add(EntityIndex.NEW_ENTITY.value)
        entity_labels[wp.idx] = (subject_entity_indices, set())
    return entity_labels
