import impl.dbpedia.util as dbp_util

NAMESPACE_DBP_LIST = dbp_util.NAMESPACE_DBP_RESOURCE + 'List_of_'


def remove_listpage_prefix(list_page: str) -> str:
    if not is_listpage(list_page):
        raise ValueError(f'Trying to remove prefix, but "{list_page}" is not a listpage.')
    return list_page[len(NAMESPACE_DBP_LIST):]


def is_listpage(obj: str) -> bool:
    return obj.startswith(NAMESPACE_DBP_LIST)
