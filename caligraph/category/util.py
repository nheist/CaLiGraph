import caligraph.util.rdf as rdf_util
from caligraph.dbpedia.util import NAMESPACE_DBP_RESOURCE

NAMESPACE_DBP_CATEGORY = NAMESPACE_DBP_RESOURCE + 'Category:'


def name2category(name: str) -> str:
    return rdf_util.name2uri(name, NAMESPACE_DBP_CATEGORY)


def category2name(resource: str) -> str:
    return rdf_util.uri2name(resource, NAMESPACE_DBP_CATEGORY)
