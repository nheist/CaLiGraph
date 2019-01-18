from . import util as list_util
import impl.category.store as cat_store
import impl.category.util as cat_util
import impl.dbpedia.store as dbp_store
import util


def get_equivalent_listpage(category: str) -> str:
    global __EQUIVALENT_LISTPAGE_MAPPING__
    if '__EQUIVALENT_LISTPAGE_MAPPING__' not in globals():
        __EQUIVALENT_LISTPAGE_MAPPING__ = util.load_or_create_cache('dbpedia_listpage_equivalents', _create_equivalent_listpage_mapping)

    return __EQUIVALENT_LISTPAGE_MAPPING__[category]


def _create_equivalent_listpage_mapping() -> dict:
    util.get_logger().info('CACHE: Creating equivalent-listpage mapping')

    categories = cat_store.get_all_cats()
    cat_to_lp_mapping = {}

    # 1) find equivalent lists by matching category/list names exactly
    name_to_category_mapping = {cat_util.remove_category_prefix(cat): cat for cat in categories}
    name_to_list_mapping = {list_util.remove_listpage_prefix(lp): lp for lp in get_listpages()}
    equal_pagenames = set(name_to_category_mapping).intersection(set(name_to_list_mapping))
    cat_to_lp_mapping.update({name_to_category_mapping[name]: name_to_list_mapping[name] for name in equal_pagenames})

    # 2) find equivalent lists by using topical concepts of categories
    for cat in categories.difference(set(cat_to_lp_mapping)):
        listpage_topics = {topic for topic in cat_store.get_topics(cat) if list_util.is_listpage(topic)}
        if len(listpage_topics) == 1:
            cat_to_lp_mapping[cat] = listpage_topics.pop()
            util.get_logger().debug(f'Mapping via topical concept: {cat} -> {cat_to_lp_mapping[cat]}')

    # 3) find equivalent lists by looking for categories containing exactly one list
    for cat in categories.difference(set(cat_to_lp_mapping)):
        listpage_members = {page for page in cat_store.get_resources(cat) if list_util.is_listpage(page)}
        if len(listpage_members) == 1:
            cat_to_lp_mapping[cat] = listpage_members.pop()
            util.get_logger().debug(f'Mapping via contained list: {cat} -> {cat_to_lp_mapping[cat]}')

    return cat_to_lp_mapping


def get_child_listpages(category: str) -> set:
    global __CHILD_LISTPAGES_MAPPING__
    if '__CHILD_LISTPAGES_MAPPING__' not in globals():
        __CHILD_LISTPAGES_MAPPING__ = util.load_or_create_cache('dbpedia_listpage_children', _create_child_listpages_mapping)

    return __CHILD_LISTPAGES_MAPPING__[category]


def _create_child_listpages_mapping() -> dict:
    util.get_logger().info('CACHE: Creating child-listpage mapping')

    # find child lists by looking for categories containing multiple lists

    pass


def get_listpages() -> set:
    global __LISTPAGES__
    if '__LISTPAGES__' not in globals():
        initializer = lambda: {res for res in dbp_store.get_resources() if list_util.is_listpage(res)}
        __LISTPAGES__ = util.load_or_create_cache('dbpedia_listpages', initializer)

    return __LISTPAGES__
