from collections import defaultdict
import util
from .nif_parser import extract_wiki_corpus_resources
from .xml_parser import _parse_raw_markup_from_xml
from .article_parser import _parse_articles
from .category_parser import _extract_parent_categories_from_markup, TEMPLATE_PREFIX, CATEGORY_PREFIX


def get_parsed_articles() -> dict:
    initializer = lambda: _parse_articles(_get_raw_articles_from_xml())
    return defaultdict(lambda: None, util.load_or_create_cache('wikipedia_parsed_articles', initializer))


def extract_parent_categories() -> dict:
    initializer = lambda: _extract_parent_categories_from_markup(_get_raw_categories_and_templates_from_xml())
    return util.load_or_create_cache('wikipedia_parent_categories', initializer)


def _get_raw_categories_and_templates_from_xml() -> tuple:
    raw_markup = _get_raw_markup_from_xml()
    categories = {name: markup for name, markup in raw_markup if CATEGORY_PREFIX in name}
    templates = {name: markup for name, markup in raw_markup if TEMPLATE_PREFIX in name}
    return categories, templates


def _get_raw_articles_from_xml() -> dict:
    return {name: markup for name, markup in _get_raw_markup_from_xml().items() if TEMPLATE_PREFIX not in name and CATEGORY_PREFIX not in name}


def _get_raw_markup_from_xml() -> dict:
    return defaultdict(str, util.load_or_create_cache('wikipedia_raw_markup', _parse_raw_markup_from_xml))
