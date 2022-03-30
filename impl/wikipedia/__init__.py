from collections import defaultdict
import utils
from .nif_parser import extract_wiki_corpus_resources
from .xml_parser import _parse_raw_markup_from_xml
from .page_parser import _parse_pages, PageType
from .category_parser import _extract_parent_categories_from_markup, TEMPLATE_PREFIX, CATEGORY_PREFIX
from impl.dbpedia.resource import DbpResourceStore, DbpResource
from typing import Dict, Tuple, Set, Optional


def get_parsed_articles() -> Dict[DbpResource, Optional[dict]]:
    initializer = lambda: _parse_pages(_get_raw_articles_from_xml())
    return defaultdict(lambda: None, utils.load_or_create_cache('wikipedia_parsed_articles', initializer))


def extract_parent_categories() -> Dict[str, Set[str]]:
    initializer = lambda: _extract_parent_categories_from_markup(_get_raw_categories_and_templates_from_xml())
    return utils.load_or_create_cache('wikipedia_parent_categories', initializer)


def _get_raw_categories_and_templates_from_xml() -> Tuple[Dict[str, str], Dict[str, str]]:
    raw_markup = _get_raw_markup_from_xml()
    categories = {name: markup for name, markup in raw_markup.items() if CATEGORY_PREFIX in name}
    templates = {name: markup for name, markup in raw_markup.items() if TEMPLATE_PREFIX in name}
    return categories, templates


def _get_raw_articles_from_xml() -> Dict[DbpResource, str]:
    dbr = DbpResourceStore.instance()
    return {dbr.get_resource_by_name(name): markup for name, markup in _get_raw_markup_from_xml().items() if dbr.has_resource_with_name(name)}


def _get_raw_markup_from_xml() -> Dict[str, str]:
    return defaultdict(str, utils.load_or_create_cache('wikipedia_raw_markup', _parse_raw_markup_from_xml))
