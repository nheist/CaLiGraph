from typing import List, Set, Tuple, Dict
from collections import defaultdict
from itertools import cycle, islice
import random
import utils
from impl.util.string import alternate_iters_to_string
from impl.util.transformer import SpecialToken, EntityIndex
from impl.dbpedia.resource import DbpResourceStore
from impl.dbpedia.category import DbpCategoryStore
from impl.wikipedia import WikiPageStore, WikiPage
from impl.wikipedia.page_parser import ListingId, MentionId, WikiListing, WikiTable, WikiListingItem, WikiEnumEntry
from impl.caligraph.entity import ClgEntityStore, ClgEntity
from .util import DataCorpus, Alignment, CXS, CXE, ROW, COL


class ListingDataCorpus(DataCorpus):
    def __init__(self, listing_ids: List[ListingId], alignment: Alignment):
        self.listing_ids = listing_ids
        self.alignment = alignment

    def get_listings(self) -> List[WikiListing]:
        wps = WikiPageStore.instance()
        return [wps.get_listing(listing_id) for listing_id in self.listing_ids]

    def get_entities(self) -> Set[ClgEntity]:
        return ClgEntityStore.instance().get_entities()

    def get_mention_labels(self, discard_unknown: bool = False) -> Dict[MentionId, str]:
        mention_labels = {}
        for listing in self.get_listings():
            for item in listing.get_items(has_subject_entity=True):
                if discard_unknown and item.subject_entity.entity_idx == EntityIndex.NEW_ENTITY:
                    continue
                mention_id = MentionId(listing.page_idx, listing.idx, item.idx)
                mention_labels[mention_id] = item.subject_entity.label
        return mention_labels

    def get_mention_input(self, add_page_context: bool, add_text_context: bool) -> Tuple[Dict[MentionId, str], Dict[MentionId, bool]]:
        utils.get_logger().debug('Preparing listing items..')
        result = {}
        result_ent_known = {}
        if not add_page_context and not add_text_context:
            for l in self.get_listings():
                for i in l.get_items(has_subject_entity=True):
                    mention_id = MentionId(l.page_idx, l.idx, i.idx)
                    se = i.subject_entity
                    result[mention_id] = f'{se.label} {SpecialToken.get_type_token(se.entity_type)}'
                    result_ent_known[mention_id] = se.entity_idx != EntityIndex.NEW_ENTITY
            return result, result_ent_known
        for listing in self.get_listings():
            prepared_context = self._prepare_listing_context(listing)
            prepared_items = [self._prepare_listing_item(item) for item in listing.get_items()]
            for idx, item in enumerate(listing.get_items(has_subject_entity=True)):
                mention_id = MentionId(listing.page_idx, listing.idx, item.idx)
                item_se = item.subject_entity
                # add subject entity, its type, and page context
                item_content = f' {CXS} '.join([f'{item_se.label} {SpecialToken.get_type_token(item_se.entity_type)}', prepared_context])
                # add item and `add_text_context` subsequent items (add items from start if no subsequent items left)
                item_content += ''.join(islice(cycle(prepared_items), idx, idx + add_text_context + 1))
                result[mention_id] = item_content
                result_ent_known[mention_id] = item_se.entity_idx != EntityIndex.NEW_ENTITY
        return result, result_ent_known

    @classmethod
    def _prepare_listing_context(cls, listing: WikiListing) -> str:
        res = DbpResourceStore.instance().get_resource_by_idx(listing.page_idx)
        # add label
        res_description = f'{res.get_label()} {SpecialToken.get_type_token(res.get_type_label())}'
        # add categories
        cats = list(DbpCategoryStore.instance().get_categories_for_resource(res.idx))[:3]
        cats_text = ' | '.join([cat.get_label() for cat in cats])
        # assemble context
        ctx = [res_description, cats_text, listing.topsection.title, listing.section.title]
        if isinstance(listing, WikiTable):
            ctx.append(cls._prepare_listing_item(listing.header))
        return f' {CXS} '.join(ctx) + f' {CXE} '

    @classmethod
    def _prepare_listing_item(cls, item: WikiListingItem) -> str:
        if isinstance(item, WikiEnumEntry):
            tokens = [SpecialToken.get_entry_by_depth(item.depth)] + item.tokens
            whitespaces = [' '] + item.whitespaces[:-1] + [' ']
        else:  # WikiTableRow
            tokens, whitespaces = [], []
            for cell_tokens, cell_whitespaces in zip(item.tokens, item.whitespaces):
                tokens += [COL] + cell_tokens
                whitespaces += [' '] + cell_whitespaces[:-1] + [' ']
            tokens[0] = ROW  # special indicator for start of table row
        return alternate_iters_to_string(tokens, whitespaces)


def _init_listing_data_corpora(sample_size: int) -> Tuple[ListingDataCorpus, ListingDataCorpus, ListingDataCorpus]:
    train_pages, val_pages, test_pages = _load_page_data(sample_size)
    return _create_corpus_from_pages(train_pages), _create_corpus_from_pages(val_pages), _create_corpus_from_pages(test_pages)


def _create_corpus_from_pages(pages: List[WikiPage]) -> ListingDataCorpus:
    listings = [listing for page in pages for listing in page.get_listings() if listing.has_subject_entities()]
    listing_ids = [listing.get_id() for listing in listings]
    entity_to_mention_mapping = defaultdict(set)
    for listing in listings:
        for item in listing.get_items(has_subject_entity=True, has_known_entity=True):
            mention_id = MentionId(listing.page_idx, listing.idx, item.idx)
            entity_to_mention_mapping[item.subject_entity.entity_idx].add(mention_id)
    clge = ClgEntityStore.instance()
    valid_entities = {ent_id for ent_id in entity_to_mention_mapping if clge.has_entity_with_idx(ent_id)}
    return ListingDataCorpus(listing_ids, Alignment(entity_to_mention_mapping, valid_entities))


def _load_page_data(sample_size: float) -> Tuple[List[WikiPage], List[WikiPage], List[WikiPage]]:
    # retrieve all pages with subject entities
    wiki_pages = [wp for wp in WikiPageStore.instance().get_pages() if wp.has_subject_entities()]
    # sample, shuffle, split
    wiki_pages = random.sample(wiki_pages, int(len(wiki_pages) * sample_size / 100))
    random.shuffle(wiki_pages)
    # divide into 60% train, 20% val, 20% test
    val_start_idx = int(len(wiki_pages) * .6)
    test_start_idx = int(len(wiki_pages) * .8)
    train_pages = wiki_pages[:val_start_idx]
    val_pages = wiki_pages[val_start_idx:test_start_idx]
    test_pages = wiki_pages[test_start_idx:]
    return train_pages, val_pages, test_pages
