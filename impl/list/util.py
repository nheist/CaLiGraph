import impl.dbpedia.util as dbp_util

NAMESPACE_DBP_LIST = dbp_util.NAMESPACE_DBP_RESOURCE + 'List_of_'


def remove_list_prefix(list_page: str) -> str:
    return list_page[len(NAMESPACE_DBP_LIST):]
