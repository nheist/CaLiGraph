"""Functionality to retrieve everything list-related from DBpedia resources."""

from . import util as list_util
import impl.dbpedia.store as dbp_store
import impl.dbpedia.pages as dbp_pages
import impl.category.store as cat_store
import util


LIST_TYPE_ENUM, LIST_TYPE_TABLE = 'list_type_enum', 'list_type_table'


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
        initializer = lambda: {(res[:res.find('__')] if '__' in res else res) for res in dbp_store.get_raw_resources() if list_util.is_listpage(res)}
        __LISTPAGES_WITH_REDIRECTS__ = util.load_or_create_cache('dbpedia_listpages', initializer)

    return __LISTPAGES_WITH_REDIRECTS__


def get_listcategories() -> set:
    """Return all list categories (i.e. categories starting with 'Lists of')."""
    global __LISTCATEGORIES__
    if '__LISTCATEGORIES__' not in globals():
        __LISTCATEGORIES__ = {lc for lc in cat_store.get_categories() if list_util.is_listcategory(lc)}

    return __LISTCATEGORIES__


def get_parsed_listpages(listpage_type: str = None) -> dict:
    """Return all list pages of the type `listpage_type` together with their parsed content."""
    global __PARSED_LISTPAGES__
    if '__PARSED_LISTPAGES__' not in globals():
        __PARSED_LISTPAGES__ = util.load_or_create_cache('dbpedia_listpage_parsed', _parse_listpages)
    return {lp: content for lp, content in __PARSED_LISTPAGES__.items() if listpage_type is None or listpage_type in content['types']}


def _parse_listpages() -> dict:
    parsed_listpages = {}
    for resource, content in dbp_pages.get_all_parsed_pages().items():
        if not list_util.is_listpage(resource):
            continue
        if resource != dbp_store.resolve_redirect(resource):
            continue
        if not content or 'sections' not in content:
            continue
        listpage_types = _get_listpage_types(content['sections'])
        parsed_listpages[resource] = {'sections': content['sections'], 'types': listpage_types}
    return parsed_listpages


def _get_listpage_types(listpage_sections: list) -> list:
    """Return layout types of the list page based on whether they contain enumeration entries and table rows."""
    listpage_types = []
    if any('enums' in section for section in listpage_sections):
        listpage_types.append(LIST_TYPE_ENUM)
    if any('tables' in section for section in listpage_sections):
        listpage_types.append(LIST_TYPE_TABLE)
    return listpage_types


def get_listpages_with_markup() -> dict:
    """Return the WikiText markup for all list pages."""
    return {page: markup for page, markup in dbp_pages.get_all_pages_markup() if list_util.is_listpage(page)}
