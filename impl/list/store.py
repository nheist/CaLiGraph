from . import util as list_util
import impl.category.store as cat_store
import util


def get_equivalent_list(category: str) -> str:
    global __EQUIVALENT_LIST_MAPPING__
    if '__EQUIVALENT_LIST_MAPPING__' not in globals():
        __EQUIVALENT_LIST_MAPPING__ = util.load_or_create_cache('dbpedia_list_equivalents', _create_equivalent_list_mapping)

    return __EQUIVALENT_LIST_MAPPING__[category]


def _create_equivalent_list_mapping() -> dict:
    categories = cat_store.get_all_cats()   

    # 1) find equivalent lists by matching category/list names exactly


    # 2) find equivalent lists by using topical concepts of categories

    # 3) find equivalent lists by looking for categories containing exactly one list

    pass


def get_child_lists(category: str) -> set:
    global __CHILD_LISTS_MAPPING__
    if '__CHILD_LISTS_MAPPING__' not in globals():
        __CHILD_LISTS_MAPPING__ = util.load_or_create_cache('dbpedia_list_children', _create_child_lists_mapping)

    return __CHILD_LISTS_MAPPING__[category]


def _create_child_lists_mapping() -> dict:
    # find child lists by looking for categories containing multiple lists

    pass
