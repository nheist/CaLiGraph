from . import util as list_util
import impl.category.store as cat_store
import impl.category.util as cat_util
import impl.dbpedia.store as dbp_store
import util
import impl.util.nlp as nlp_util


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
    name_to_category_mapping = {cat_util.remove_category_prefix(cat).lower(): cat for cat in categories}
    name_to_list_mapping = {list_util.remove_listpage_prefix(lp).lower(): lp for lp in get_listpages()}
    equal_pagenames = set(name_to_category_mapping).intersection(set(name_to_list_mapping))
    cat_to_lp_mapping.update({name_to_category_mapping[name]: name_to_list_mapping[name] for name in equal_pagenames})
    util.get_logger().debug(f'Found {len(cat_to_lp_mapping)} equivalent matches.')

    # 2) find equivalent lists by using topical concepts of categories and categories containing exactly one list
    for cat in categories.difference(set(cat_to_lp_mapping)):
        # topical concepts
        candidates = [topic for topic in cat_store.get_topics(cat) if list_util.is_listpage(topic)]
        # categories with exactly one list
        listpage_members = {page for page in cat_store.get_resources(cat) if list_util.is_listpage(page)}
        if len(listpage_members) == 1:
            lp = listpage_members.pop()
            if lp not in candidates:
                candidates.append(lp)

        cat_lemmas = nlp_util.filter_important_words(cat_util.category2name(cat))
        for lp in candidates:
            listpage_lemmas = nlp_util.filter_important_words(list_util.list2name(lp))
            if cat_lemmas == listpage_lemmas:
                cat_to_lp_mapping[cat] = lp
                util.get_logger().debug(f'Mapping: {cat} -> {lp}')
                break
            else:
                util.get_logger().debug(f'No map: {cat} -> {lp}')
    util.get_logger().debug(f'Found {len(cat_to_lp_mapping)} overall matches.')

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
