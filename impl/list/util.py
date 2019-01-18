import impl.dbpedia.util as dbp_util
import impl.util.rdf as rdf_util

NAMESPACE_DBP_LIST = dbp_util.NAMESPACE_DBP_RESOURCE + 'List_of_'


def remove_listpage_prefix(listpage: str) -> str:
    if not is_listpage(listpage):
        raise ValueError(f'Trying to remove prefix, but "{listpage}" is not a listpage.')
    return listpage[len(NAMESPACE_DBP_LIST):]


def is_listpage(obj: str) -> bool:
    return obj.startswith(NAMESPACE_DBP_LIST)


def list2name(listpage):
    return rdf_util.uri2name(listpage, NAMESPACE_DBP_LIST)
