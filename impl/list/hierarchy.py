from . import util as list_util
import impl.category.store as cat_store
import impl.category.util as cat_util
import impl.category.wikitaxonomy as cat_wikitax
from impl.category.conceptual import is_conceptual_category
import impl.dbpedia.store as dbp_store
import impl.list.store as list_store
import util
import impl.util.nlp as nlp_util
from collections import defaultdict
from spacy.tokens import Doc


def get_equivalent_category(listpage: str) -> str:
    global __EQUIVALENT_CATEGORY_MAPPING__
    if '__EQUIVALENT_CATEGORY_MAPPING__' not in globals():
        __EQUIVALENT_CATEGORY_MAPPING__ = defaultdict(lambda: None)
        get_equivalent_listpage('')  # initialise equivalent-listpage mapping
        for cat, lp in __EQUIVALENT_LISTPAGE_MAPPING__.items():
            __EQUIVALENT_CATEGORY_MAPPING__[lp] = cat

    return __EQUIVALENT_CATEGORY_MAPPING__[listpage]


def get_equivalent_listpage(category: str) -> str:
    global __EQUIVALENT_LISTPAGE_MAPPING__
    if '__EQUIVALENT_LISTPAGE_MAPPING__' not in globals():
        __EQUIVALENT_LISTPAGE_MAPPING__ = defaultdict(lambda: None, util.load_or_create_cache('dbpedia_listpage_equivalents', _create_equivalent_listpage_mapping))

    return __EQUIVALENT_LISTPAGE_MAPPING__[category]


def _create_equivalent_listpage_mapping() -> dict:
    util.get_logger().info('CACHE: Creating equivalent-listpage mapping')
    categories = cat_store.get_all_cats()
    cat_to_lp_mapping = {}

    # 1) find equivalent lists by matching category/list names exactly
    name_to_category_mapping = {cat_util.remove_category_prefix(cat).lower(): cat for cat in categories}
    name_to_list_mapping = {list_util.remove_listpage_prefix(lp).lower(): lp for lp in list_store.get_listpages_with_redirects()}
    equal_pagenames = set(name_to_category_mapping).intersection(set(name_to_list_mapping))
    cat_to_lp_mapping.update({name_to_category_mapping[name]: name_to_list_mapping[name] for name in equal_pagenames})

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

        # find approximate matches of category and listpage
        cat_lemmas = nlp_util.filter_important_words(cat_util.category2name(cat))
        identified_lps = set()
        for lp in candidates:
            listpage_lemmas = nlp_util.filter_important_words(list_util.list2name(lp))
            if len(cat_lemmas) == len(listpage_lemmas):
                if all(any(nlp_util.is_synonym(ll, cl) for ll in listpage_lemmas) for cl in cat_lemmas):
                    if all(any(nlp_util.is_synonym(ll, cl) for cl in cat_lemmas) for ll in listpage_lemmas):
                        identified_lps.add(lp)
        # only accept them as match if there is a 1:1 relationship
        if len(identified_lps) == 1:
            identified_lp = identified_lps.pop()
            if identified_lp not in cat_to_lp_mapping.values():
                cat_to_lp_mapping[cat] = identified_lp

    # 3) resolve all redirects of listpages and only return those that point to listpages
    return {cat: dbp_store.resolve_redirect(lp) for cat, lp in cat_to_lp_mapping.items() if list_util.is_listpage(dbp_store.resolve_redirect(lp))}


def get_parent_categories(listpage: str) -> set:
    global __PARENT_CATEGORIES_MAPPING__
    if '__PARENT_CATEGORIES_MAPPING__' not in globals():
        __PARENT_CATEGORIES_MAPPING__ = defaultdict(set)
        get_child_listpages('')  # initialise child-listpage mapping
        for cat, lps in __CHILD_LISTPAGES_MAPPING__.items():
            for lp in lps:
                __PARENT_CATEGORIES_MAPPING__[lp].add(cat)

    return __PARENT_CATEGORIES_MAPPING__[listpage]


def get_child_listpages(category: str) -> set:
    global __CHILD_LISTPAGES_MAPPING__
    if '__CHILD_LISTPAGES_MAPPING__' not in globals():
        __CHILD_LISTPAGES_MAPPING__ = util.load_or_create_cache('dbpedia_listpage_children', _create_child_listpages_mapping)

    return __CHILD_LISTPAGES_MAPPING__[category]


def _create_child_listpages_mapping() -> dict:
    util.get_logger().info('CACHE: Creating child-listpage mapping')

    cat_to_lp_mapping = defaultdict(set)
    for lp in list_store.get_listpages():
        if get_equivalent_category(lp):
            continue

        headlemmas = nlp_util.get_head_lemmas(nlp_util.parse(list_util.list2name(lp)))

        lp_cats = cat_store.get_resource_to_cats_mapping()[lp]
        # check if headlemma of lp matches with cat. if yes -> add
        parent_category_docs = {cat: nlp_util.parse(cat_util.category2name(cat)) for cat in lp_cats if not list_util.is_listcategory(cat)}
        matching_cats = _find_cats_with_matching_headlemmas(parent_category_docs, headlemmas)
        if matching_cats:
            for cat in matching_cats:
                cat_to_lp_mapping[cat].add(lp)
            continue

        # TODO: implement instance-based category retrieval (only if listpage has the category)

        # check if headlemma of lp matches with lists-of cat. if yes -> check for lists-of cat hierarchy path. if good -> add
        parent_listcategory_docs = {cat: nlp_util.parse(list_util.listcategory2name(cat)) for cat in lp_cats if list_util.is_listcategory(cat)}
        if len(parent_listcategory_docs) > 1:
            # if we have more than one listcategory, we only use those with a matching headlemma
            parent_listcategory_docs = {cat: parent_listcategory_docs[cat] for cat in _find_cats_with_matching_headlemmas(parent_listcategory_docs, headlemmas)}

        for listcat, doc in parent_listcategory_docs.items():
            parent_categories = _find_listcategory_hierarchy(listcat, doc)[-1]
            for cat in parent_categories:
                cat_to_lp_mapping[cat].add(lp)

    return cat_to_lp_mapping


def _find_cats_with_matching_headlemmas(category_docs: dict, headlemmas: set):
    matches = set()
    for cat, cat_doc in category_docs.items():
        cat_headlemmas = nlp_util.get_head_lemmas(cat_doc)
        if any(cat_wikitax.is_hypernym(chl, hl) for chl in cat_headlemmas for hl in headlemmas):
            matches.add(cat)
    return matches


def _find_listcategory_hierarchy(listcategory: str, listcat_doc: Doc):
    headlemmas = nlp_util.get_head_lemmas(listcat_doc)

    parent_cats = cat_store.get_parents(listcategory)
    # check parent categories for same head lemmas
    parent_category_docs = {cat: nlp_util.parse(cat_util.category2name(cat)) for cat in parent_cats if not list_util.is_listcategory(cat)}
    matching_cats = {cat for cat in _find_cats_with_matching_headlemmas(parent_category_docs, headlemmas) if is_conceptual_category(cat)}
    if matching_cats:
        return [listcategory, set(matching_cats)]

    # check parent listcategories for same head lemmas and propagate search
    parent_listcategory_docs = {cat: nlp_util.parse(list_util.listcategory2name(cat)) for cat in parent_cats if list_util.is_listcategory(cat)}
    parent_listcategory_docs = {cat: parent_listcategory_docs[cat] for cat in _find_cats_with_matching_headlemmas(parent_listcategory_docs, headlemmas)}
    if parent_listcategory_docs:
        return [listcategory] + _find_listcategory_hierarchy(*parent_listcategory_docs.popitem())

    # return empty parent category as no link to the existing category hierarchy can be found
    return [listcategory, set()]
