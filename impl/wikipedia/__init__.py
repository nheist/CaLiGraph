from collections import defaultdict
import utils
from .nif_parser import extract_wiki_corpus_resources
from .xml_parser import _parse_raw_markup_from_xml
from .page_parser import _parse_pages, PageType
from .category_parser import _extract_parent_categories_from_markup
from impl.dbpedia.resource import DbpResourceStore, DbpResource, DbpFile
from typing import Dict, Tuple, Set, Optional
from impl.util.rdf import Namespace


def get_parsed_pages() -> Dict[DbpResource, Optional[dict]]:
    initializer = lambda: _parse_pages(_get_raw_pages_from_xml())
    return utils.load_or_create_cache('wikipedia_parsed_pages', initializer)


def extract_parent_categories() -> Dict[str, Set[str]]:
    initializer = lambda: _extract_parent_categories_from_markup(_get_raw_categories_and_templates_from_xml())
    return utils.load_or_create_cache('wikipedia_parent_categories', initializer)


def _get_raw_categories_and_templates_from_xml() -> Tuple[Dict[str, str], Dict[str, str]]:
    raw_markup = _get_raw_markup_from_xml()
    categories = {name: markup for name, markup in raw_markup.items() if Namespace.PREFIX_CATEGORY.value in name}
    templates = {name: markup for name, markup in raw_markup.items() if Namespace.PREFIX_TEMPLATE.value in name}
    return categories, templates


def _get_raw_pages_from_xml() -> Dict[DbpResource, str]:
    dbr = DbpResourceStore.instance()
    pages = {dbr.get_resource_by_iri(iri): markup for iri, markup in _get_raw_markup_from_xml().items() if dbr.has_resource_with_iri(iri)}
    return {res: page_content for res, page_content in pages.items() if not res.is_meta and not isinstance(res, DbpFile)}


def _get_raw_markup_from_xml() -> Dict[str, str]:
    return defaultdict(str, utils.load_or_create_cache('wikipedia_raw_markup', _parse_raw_markup_from_xml))
