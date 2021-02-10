"""Extract entities from listings in arbitrary Wikipedia pages.
Learn association rules to gather types and relations for the extracted entities."""


import utils
from .extract import _extract_page_entities


def get_page_entities(graph) -> dict:
    global __PAGE_ENTITES__
    if '__PAGE_ENTITES__' not in globals():
        __PAGE_ENTITES__ = utils.load_or_create_cache('listing_page_entities', lambda: _extract_page_entities(graph))
    return __PAGE_ENTITES__
