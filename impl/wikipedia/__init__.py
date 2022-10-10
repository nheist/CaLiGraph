from impl.util.singleton import Singleton
from typing import Dict, Tuple, Set, List, Iterable
from collections import defaultdict
import utils
from .nif_parser import extract_wiki_corpus_resources
from .xml_parser import _parse_raw_markup_from_xml
from .page_parser import _parse_pages, WikiPage, WikiSubjectEntity
from .category_parser import _extract_parent_categories_from_markup
from impl.dbpedia.resource import DbpListpage
import impl.dbpedia.util as dbp_util
from impl.util.rdf import Namespace
from impl.util.nlp import EntityTypeLabel
from impl.util.transformer import EntityIndex


@Singleton
class WikiPageStore:
    def __init__(self):
        self.pages = {p.idx: p for p in self._init_page_cache()}

    def _init_page_cache(self) -> List[WikiPage]:
        return utils.load_or_create_cache('wikipedia_parsed_pages', lambda: _parse_pages(_get_raw_pages_from_xml()))

    def get_page(self, page_idx: int) -> WikiPage:
        return self.pages[page_idx]

    def get_pages(self) -> Iterable[WikiPage]:
        return self.pages.values()

    def get_listpages(self) -> Iterable[WikiPage]:
        return [p for p in self.get_pages() if isinstance(p.resource, DbpListpage)]

    def get_subject_entity(self, item_id: Tuple[int, int, int]) -> WikiSubjectEntity:
        page_idx, listing_idx, item_idx = item_id
        return self.pages[page_idx].listings[listing_idx].items[item_idx].subject_entity

    def set_subject_entity_mentions(self, subject_entity_mentions: Dict[int, Dict[int, Dict[int, Tuple[str, EntityTypeLabel]]]]):
        self._reset_subject_entities()
        for wp_idx, wp_mentions in subject_entity_mentions.items():
            wp = self.pages[wp_idx]
            for listing_idx, listing_mentions in wp_mentions.items():
                listing = wp.listings[listing_idx]
                for item_idx, (se_label, se_type) in listing_mentions.items():
                    item = listing.items[item_idx]
                    # check whether there is an existing mention for the subject entity -> if yes, use its entity_idx
                    se_idx = EntityIndex.NEW_ENTITY.value
                    for mention in item.get_mentions():
                        if mention.label == se_label:
                            se_idx = mention.entity_idx
                            break
                    item.subject_entity = WikiSubjectEntity(se_idx, se_label, se_type)

    def _reset_subject_entities(self):
        for wp in self.get_pages():
            for listing in wp.get_listings():
                for item in listing.get_items():
                    item.subject_entity = None

    def add_disambiguated_subject_entities(self, disambiguated_subject_entities: Dict[int, Dict[int, Dict[int, int]]]):
        for wp_idx, wp_entities in disambiguated_subject_entities.items():
            wp = self.pages[wp_idx]
            for listing_idx, listing_entities in wp_entities.items():
                listing = wp.listings[listing_idx]
                for item_idx, se_idx in listing_entities.items():
                    listing.items[item_idx].subject_entity.entity_idx = se_idx


def extract_parent_categories() -> Dict[str, Set[str]]:
    initializer = lambda: _extract_parent_categories_from_markup(_get_raw_categories_and_templates_from_xml())
    return utils.load_or_create_cache('wikipedia_parent_categories', initializer)


def _get_raw_categories_and_templates_from_xml() -> Tuple[Dict[str, str], Dict[str, str]]:
    raw_markup = _get_raw_markup_from_xml()
    categories = {name: markup for name, markup in raw_markup.items() if Namespace.PREFIX_CATEGORY.value in name}
    templates = {name: markup for name, markup in raw_markup.items() if Namespace.PREFIX_TEMPLATE.value in name}
    return categories, templates


def _get_raw_pages_from_xml() -> Dict[str, str]:
    return {dbp_util.resource_iri2name(iri): markup for iri, markup in _get_raw_markup_from_xml().items()}


def _get_raw_markup_from_xml() -> Dict[str, str]:
    return defaultdict(str, utils.load_or_create_cache('wikipedia_raw_markup', _parse_raw_markup_from_xml))
