"""Utilities to work with list pages and list categories."""

import impl.dbpedia.util as dbp_util
import impl.util.rdf as rdf_util

NAMESPACE_DBP_LIST = dbp_util.NAMESPACE_DBP_RESOURCE + 'List_of_'
NAMESPACE_DBP_LISTS = dbp_util.NAMESPACE_DBP_RESOURCE + 'Lists_of_'
NAMESPACE_DBP_LISTCAT = dbp_util.NAMESPACE_DBP_RESOURCE + 'Category:Lists_of_'


def is_listpage(obj: str) -> bool:
    return obj.startswith(NAMESPACE_DBP_LIST)


def is_listspage(obj: str) -> bool:
    return obj.startswith(NAMESPACE_DBP_LISTS)


def listpage2name(lp: str) -> str:
    if is_listpage(lp):
        return rdf_util.uri2name(lp, NAMESPACE_DBP_LIST)
    return rdf_util.uri2name(lp, NAMESPACE_DBP_LISTS)


def is_listcategory(obj: str) -> bool:
    return obj.startswith(NAMESPACE_DBP_LISTCAT)


def listcategory2name(listcategory: str) -> str:
    return rdf_util.uri2name(listcategory, NAMESPACE_DBP_LISTCAT)


def list2name(lst: str) -> str:
    if is_listcategory(lst):
        return listcategory2name(lst)
    return listpage2name(lst)
