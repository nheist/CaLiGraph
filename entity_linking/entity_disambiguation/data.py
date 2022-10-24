from typing import List, Set, Tuple, Union, NamedTuple
from collections import defaultdict
import itertools
import random
from impl.wikipedia import WikiPageStore, WikiPage
from impl.wikipedia.page_parser import ListingId, MentionId, WikiListing
from impl.caligraph.entity import ClgEntityStore, ClgEntity


class Pair(NamedTuple):
    source: MentionId
    target: Union[MentionId, int]
    confidence: float

    def __eq__(self, other) -> bool:
        return self.source == other.source and self.target == other.target

    def __hash__(self):
        return self.source.__hash__() + self.target.__hash__()


class DataCorpus:
    def __init__(self, listing_ids: List[ListingId], mm_alignment: Set[Pair], me_alignment: Set[Pair]):
        self.listing_ids = listing_ids
        self.mm_alignment = mm_alignment
        self.me_alignment = me_alignment

    def get_listings(self) -> List[WikiListing]:
        wps = WikiPageStore.instance()
        return [wps.get_listing(listing_id) for listing_id in self.listing_ids]

    def get_entities(self) -> Set[ClgEntity]:
        return ClgEntityStore.instance().get_entities()


def get_train_val_test_corpora() -> Tuple[DataCorpus, DataCorpus, DataCorpus]:
    train_pages, val_pages, test_pages = _load_page_data(.05)
    return _create_corpus_from_pages(train_pages), _create_corpus_from_pages(val_pages), _create_corpus_from_pages(test_pages)


def _create_corpus_from_pages(pages: List[WikiPage]) -> DataCorpus:
    listings = [listing for page in pages for listing in page.get_listings() if listing.has_subject_entities()]
    entity_to_mention_mapping = defaultdict(set)
    for listing in listings:
        for item in listing.get_items(has_subject_entity=True, has_known_entity=True):
            mention_id = MentionId(listing.page_idx, listing.idx, item.idx)
            entity_to_mention_mapping[item.subject_entity.entity_idx].add(mention_id)
    mm_alignment = set()
    for mention_group in entity_to_mention_mapping.values():
        mm_alignment.update({Pair(*sorted(item_pair), 1) for item_pair in itertools.combinations(mention_group, 2)})
    me_alignment = set()
    clge = ClgEntityStore.instance()
    for ent_id, mentions in entity_to_mention_mapping.items():
        if not clge.has_entity_with_idx(ent_id):
            continue  # discard unknown entities
        me_alignment.update({Pair(mention_id, ent_id, 1) for mention_id in mentions})
    return DataCorpus([listing.get_id() for listing in listings], mm_alignment, me_alignment)


def _load_page_data(sample_size: float) -> Tuple[List[WikiPage], List[WikiPage], List[WikiPage]]:
    # retrieve all pages with subject entities
    wiki_pages = [wp for wp in WikiPageStore.instance().get_pages() if wp.has_subject_entities()]
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
