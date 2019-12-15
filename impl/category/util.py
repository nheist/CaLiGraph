"""Utilities to work with categories."""

import impl.util.rdf as rdf_util
from impl.dbpedia.util import NAMESPACE_DBP_RESOURCE

NAMESPACE_DBP_CATEGORY = NAMESPACE_DBP_RESOURCE + 'Category:'


def name2category(name: str) -> str:
    return rdf_util.name2uri(name, NAMESPACE_DBP_CATEGORY)


def category2name(category: str) -> str:
    return rdf_util.uri2name(category, NAMESPACE_DBP_CATEGORY)


def is_category(obj: str) -> bool:
    return obj.startswith(NAMESPACE_DBP_CATEGORY)


def remove_category_prefix(category: str) -> str:
    return category[len(NAMESPACE_DBP_CATEGORY):]
