from . import util as list_util
import impl.category.base as cat_base
import impl.category.store as cat_store
import impl.category.util as cat_util
from impl.category.conceptual import is_conceptual_category
import impl.dbpedia.store as dbp_store
import impl.list.store as list_store
import util
import impl.util.nlp as nlp_util
import impl.util.hypernymy as hypernymy_util
from collections import defaultdict
from spacy.tokens import Doc


"""Mapping of listpages to the category-lists-hierarchy

A mapping can have one of two types:
- Equivalence Mapping: The entities of the listpage are put directly into the mapped hierarchy item
- Child Mapping: The listpage is appended to the hierarchy as a child of an existing hierarchy item
"""


def get_equivalent_categories(listpage: str) -> set:
    global __EQUIVALENT_CATEGORY_MAPPING__
    if '__EQUIVALENT_CATEGORY_MAPPING__' not in globals():
        __EQUIVALENT_CATEGORY_MAPPING__ = defaultdict(set)
        get_equivalent_listpages('')  # make sure that equivalent-listpage mapping is initialised
        for cat, lps in __EQUIVALENT_LISTPAGE_MAPPING__.items():
            for lp in lps:
                __EQUIVALENT_CATEGORY_MAPPING__[lp].add(cat)

    return __EQUIVALENT_CATEGORY_MAPPING__[listpage]


def get_equivalent_listpages(category: str) -> set:
    global __EQUIVALENT_LISTPAGE_MAPPING__
    if '__EQUIVALENT_LISTPAGE_MAPPING__' not in globals():
        __EQUIVALENT_LISTPAGE_MAPPING__ = defaultdict(set, util.load_or_create_cache('dbpedia_listpage_equivalents', _create_equivalent_listpage_mapping))

    return __EQUIVALENT_LISTPAGE_MAPPING__[category]


def _create_equivalent_listpage_mapping() -> dict:
    util.get_logger().info('CACHE: Creating equivalent-listpage mapping')
    cat_graph = cat_base.get_cyclefree_wikitaxonomy_graph()

    # 1) find equivalent lists by matching category/list names exactly
    name_to_category_mapping = {nlp_util.remove_by_phrase_from_text(cat_util.category2name(cat)).lower(): cat for cat in cat_graph.nodes}
    name_to_list_mapping = defaultdict(set)
    for lp in list_store.get_listpages_with_redirects():
        name_to_list_mapping[nlp_util.remove_by_phrase_from_text(list_util.list2name(lp)).lower()].add(lp)
    equal_pagenames = set(name_to_category_mapping).intersection(set(name_to_list_mapping))
    cat_to_lp_name_mapping = defaultdict(set, {name_to_category_mapping[name]: name_to_list_mapping[name] for name in equal_pagenames})
    util.get_logger().debug(f'Name Mapping: Mapped {len(cat_to_lp_name_mapping)} categories to {sum(len(lps) for lps in cat_to_lp_name_mapping.values())} lists.')

    # 2) find equivalent lists by using topical concepts and members of categories
    cat_to_lp_member_mapping = defaultdict(set)

    for cat in cat_graph.nodes:
        # topical concepts
        candidates = {topic for topic in cat_store.get_topics(cat) if list_util.is_listpage(topic)}
        # category members
        candidates.update({page for page in cat_store.get_resources(cat) if list_util.is_listpage(page)})
        # find approximate matches of category and listpage
        cat_important_words = nlp_util.filter_important_words(cat_util.category2name(cat))
        mapped_lps = set()
        for lp in candidates:
            lp_important_words = nlp_util.filter_important_words(list_util.list2name(lp))
            if len(cat_important_words) == len(lp_important_words):
                if all(any(hypernymy_util.is_synonym(ll, cl) for ll in lp_important_words) for cl in cat_important_words):
                    if all(any(hypernymy_util.is_synonym(ll, cl) for cl in cat_important_words) for ll in lp_important_words):
                        mapped_lps.add(lp)
        if mapped_lps:
            cat_to_lp_member_mapping[cat] = mapped_lps
    util.get_logger().debug(f'Member Mapping: Mapped {len(cat_to_lp_member_mapping)} categories to {sum(len(lps) for lps in cat_to_lp_member_mapping.values())} lists.')

    # 3) Merge mappings and resolve redirects
    mapped_cats = set(cat_to_lp_name_mapping) | set(cat_to_lp_member_mapping)
    # join mappings
    cat_to_lp_equivalence_mapping = {cat: cat_to_lp_name_mapping[cat] | cat_to_lp_member_mapping[cat] for cat in mapped_cats}
    # resolve redirects
    cat_to_lp_equivalence_mapping = {cat: {dbp_store.resolve_redirect(lp) for lp in lps if list_util.is_listpage(dbp_store.resolve_redirect(lp))} for cat, lps in cat_to_lp_equivalence_mapping.items()}
    # remove empty mappings
    cat_to_lp_equivalence_mapping = {cat: lps for cat, lps in cat_to_lp_equivalence_mapping.items() if lps}
    util.get_logger().debug(f'Merged Equivalence Mapping: Mapped {len(cat_to_lp_equivalence_mapping)} categories to {sum(len(lps) for lps in cat_to_lp_equivalence_mapping.values())} lists.')

    return cat_to_lp_equivalence_mapping


# TODO: REWORK (methods, logging, ..)
def get_parent_categories(listpage: str) -> set:
    global __PARENT_CATEGORIES_MAPPING__
    if '__PARENT_CATEGORIES_MAPPING__' not in globals():
        __PARENT_CATEGORIES_MAPPING__ = defaultdict(set)
        get_child_listpages('')  # make sure that parent-listpage mapping is initialised
        for cat, lps in __CHILD_LISTPAGES_MAPPING__.items():
            for lp in lps:
                __PARENT_CATEGORIES_MAPPING__[lp].add(cat)

    return __PARENT_CATEGORIES_MAPPING__[listpage]


def get_child_listpages(category: str) -> set:
    global __CHILD_LISTPAGES_MAPPING__
    if '__CHILD_LISTPAGES_MAPPING__' not in globals():
        __CHILD_LISTPAGES_MAPPING__ = defaultdict(set, util.load_or_create_cache('dbpedia_listpage_children', _create_child_listpages_mapping))

    return __CHILD_LISTPAGES_MAPPING__[category]


# TODO: create lists-of hierarchy externally and then use it here
# TODO: make sure that we have an M:N mapping
# TODO: split individual mappings and provide logging infos
def _create_child_listpages_mapping() -> dict:
    util.get_logger().info('CACHE: Creating child-listpage mapping')

    cat_to_lp_mapping = defaultdict(set)
    for lp in list_store.get_listpages():
        if get_equivalent_categories(lp):
            continue

        headlemmas = nlp_util.get_head_lemmas(nlp_util.parse(list_util.list2name(lp)))

        lp_listcats = {cat for cat in cat_store.get_resource_categories(lp) if list_util.is_listcategory(cat)}
        lp_cats = cat_store.get_resource_categories(lp).difference(lp_listcats)
        # check if headlemma of lp matches with cat. if yes -> add
        parent_category_docs = {cat: nlp_util.parse(cat_util.category2name(cat)) for cat in lp_cats}
        matching_cats = _find_cats_with_matching_headlemmas(parent_category_docs, headlemmas)
        if matching_cats:
            for cat in matching_cats:
                cat_to_lp_mapping[cat].add(lp)
            continue

        # check if headlemma of lp matches with lists-of cat. if yes -> check for lists-of cat hierarchy path. if good -> add
        parent_listcategory_docs = {cat: nlp_util.parse(list_util.listcategory2name(cat)) for cat in lp_listcats}
        if len(parent_listcategory_docs) > 1:
            # if we have more than one listcategory, we only use those with a matching headlemma
            parent_listcategory_docs = {cat: parent_listcategory_docs[cat] for cat in _find_cats_with_matching_headlemmas(parent_listcategory_docs, headlemmas)}

        for listcat, doc in parent_listcategory_docs.items():
            parent_categories = _find_listcategory_hierarchy(listcat, doc)[-1]
            for cat in parent_categories:
                cat_to_lp_mapping[cat].add(lp)

    # TODO: Implement remaining actions (listpage lemma search)

    # TODO: resolve redirects and remove empty mappings // merge mappings
    return cat_to_lp_mapping


def _find_cats_with_matching_headlemmas(category_docs: dict, headlemmas: set):
    matches = set()
    for cat, cat_doc in category_docs.items():
        cat_headlemmas = nlp_util.get_head_lemmas(cat_doc)
        if any(hypernymy_util.is_hypernym(chl, hl) for chl in cat_headlemmas for hl in headlemmas):
            matches.add(cat)
    return matches


def _find_listcategory_hierarchy(listcategory: str, listcat_doc: Doc) -> list:
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
