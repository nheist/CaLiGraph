"""Functionality to retrieve everything list-related from DBpedia resources."""

import impl.listpage.util as list_util
from impl import wikipedia
import impl.dbpedia.store as dbp_store
import impl.category.store as cat_store


def get_listpages() -> set:
    """Return all list pages (with already resolved redirects)."""
    global __LISTPAGES__
    if '__LISTPAGES__' not in globals():
        __LISTPAGES__ = {dbp_store.resolve_redirect(lp) for lp in get_listpages_with_redirects() if list_util.is_listpage(dbp_store.resolve_redirect(lp))}

    return __LISTPAGES__


def get_listpages_with_redirects() -> set:
    """Return all list pages."""
    global __LISTPAGES_WITH_REDIRECTS__
    if '__LISTPAGES_WITH_REDIRECTS__' not in globals():
        __LISTPAGES_WITH_REDIRECTS__ = {(res[:res.find('__')] if '__' in res else res) for res in dbp_store.get_raw_resources() if list_util.is_listpage(res)}

    return __LISTPAGES_WITH_REDIRECTS__


def get_listcategories() -> set:
    """Return all list categories (i.e. categories starting with 'Lists of')."""
    global __LISTCATEGORIES__
    if '__LISTCATEGORIES__' not in globals():
        __LISTCATEGORIES__ = {lc for lc in cat_store.get_categories(include_listcategories=True) if list_util.is_listcategory(lc)}

    return __LISTCATEGORIES__


def get_parsed_listpages(listpage_type: str = None) -> dict:
    """Return all list pages of the type `listpage_type` together with their parsed content."""
    global __PARSED_LISTPAGES__
    if '__PARSED_LISTPAGES__' not in globals():
        __PARSED_LISTPAGES__ = _parse_listpages()
    return {lp: content for lp, content in __PARSED_LISTPAGES__.items() if listpage_type is None or listpage_type in content['types']}


def _parse_listpages() -> dict:
    parsed_listpages = {}
    for resource, content in wikipedia.get_parsed_articles().items():
        if not list_util.is_listpage(resource):
            continue
        if resource != dbp_store.resolve_redirect(resource):
            continue
        if not content or 'sections' not in content:
            continue
        parsed_listpages[resource] = content
    return parsed_listpages
