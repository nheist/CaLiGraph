from typing import List, Tuple
from collections import namedtuple, defaultdict
import itertools
import random
import utils
from impl.util.transformer import EntityIndex
from impl.wikipedia import WikiPageStore, WikiPage
from impl.caligraph.entity import ClgEntityStore


Pair = namedtuple('Pair', ['source', 'target', 'confidence'])
DataCorpus = namedtuple('DataCorpus', ['source', 'target', 'alignment'])


# MENTION-MENTION

def get_mm_train_val_test_corpora() -> Tuple[DataCorpus, DataCorpus, DataCorpus]:
    train_pages, val_pages, test_pages = _get_mm_train_val_test_pages()
    return _create_mm_corpus_from_pages(train_pages), _create_mm_corpus_from_pages(val_pages), _create_mm_corpus_from_pages(test_pages)


def _get_mm_train_val_test_pages() -> Tuple[List[WikiPage], List[WikiPage], List[WikiPage]]:
    train_idxs, val_idxs, test_idxs = utils.load_or_create_cache('MM_data', lambda: _load_page_data(.1))
    return _get_pages_for_idxs(train_idxs), _get_pages_for_idxs(val_idxs), _get_pages_for_idxs(test_idxs)


def _create_mm_corpus_from_pages(pages: List[WikiPage]) -> DataCorpus:
    listings = [listing for page in pages for listing in page.get_listings() if listing.has_subject_entities()]
    entity_to_item_mapping = defaultdict(set)
    for listing in listings:
        for item in listing.get_items():
            if not item.subject_entity or item.subject_entity.entity_idx == EntityIndex.NEW_ENTITY.value:
                continue
            entity_to_item_mapping[item.subject_entity.entity_idx].add((listing.page_idx, listing.idx, item.idx))
    alignment = set()
    for item_group in entity_to_item_mapping.values():
        alignment.update({Pair(*sorted(item_pair), 1) for item_pair in itertools.combinations(item_group, 2)})
    return DataCorpus(listings, None, alignment)


# MENTION-ENTITY

def get_me_train_val_test_corpora() -> Tuple[DataCorpus, DataCorpus, DataCorpus]:
    train_pages, val_pages, test_pages = _get_me_train_val_test_pages()
    return _create_me_corpus_from_pages(train_pages), _create_me_corpus_from_pages(val_pages), _create_me_corpus_from_pages(test_pages)


def _create_me_corpus_from_pages(pages: List[WikiPage]) -> DataCorpus:
    listings = [listing for page in pages for listing in page.get_listings() if listing.has_subject_entities()]
    clge = ClgEntityStore.instance()
    entities = clge.get_entities()
    alignment = set()
    for listing in listings:
        for item in listing.get_items():
            if not item.subject_entity or item.subject_entity.entity_idx == EntityIndex.NEW_ENTITY.value:
                continue
            item_id = (listing.page_idx, listing.idx, item.idx)
            se_id = item.subject_entity.entity_idx
            if not clge.has_entity_with_idx(se_id):
                continue
            alignment.add(Pair(item_id, se_id, 1))
    return DataCorpus(listings, entities, alignment)


def _get_me_train_val_test_pages() -> Tuple[List[WikiPage], List[WikiPage], List[WikiPage]]:
    train_idxs, val_idxs, test_idxs = utils.load_or_create_cache('ME_data', lambda: _load_page_data(.1))
    return _get_pages_for_idxs(train_idxs), _get_pages_for_idxs(val_idxs), _get_pages_for_idxs(test_idxs)


# SHARED

def _load_page_data(sample_size: float) -> Tuple[List[int], List[int], List[int]]:
    # retrieve all pages with subject entities
    wiki_pages = [wp.idx for wp in WikiPageStore.instance().get_pages() if wp.has_subject_entities()]
    # sample, shuffle, split
    wiki_pages = random.sample(wiki_pages, int(len(wiki_pages) * sample_size))
    random.shuffle(wiki_pages)
    # divide into 60% train, 20% val, 20% test
    val_start_idx = int(len(wiki_pages) * .6)
    test_start_idx = int(len(wiki_pages) * .8)
    train_pages = wiki_pages[:val_start_idx]
    val_pages = wiki_pages[val_start_idx:test_start_idx]
    test_pages = wiki_pages[test_start_idx:]
    return train_pages, val_pages, test_pages


def _get_pages_for_idxs(page_idxs: List[int]) -> List[WikiPage]:
    wps = WikiPageStore.instance()
    return [wps.get_page(idx) for idx in page_idxs]
