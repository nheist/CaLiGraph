"""Mapping of listpages to the category-hierarchy

A mapping can have one of two types:
- Equivalence Mapping: The entities of the listpage are put directly into the mapped hierarchy item, if:
  - the name of the list is equivalent to the name of a category
  - every word in the list's name is synonymous to a word in the category's name (and vice versa) [reduce complexity?]
- Child Mapping: The listpage is appended to the hierarchy as a child of an existing hierarchy item, if:
  - the headlemmas of a parent category are synonyms/hypernyms of the headlemmas of the list
  - the only parent of the list is root -> append it to root
"""

import impl.category.base as cat_base
import impl.category.store as cat_store
import impl.list.util as list_util
import impl.list.base as list_base
import util
import impl.util.nlp as nlp_util
import impl.util.hypernymy as hypernymy_util
from collections import defaultdict
import inflection


def get_equivalent_categories(lst: str) -> set:
    global __EQUIVALENT_CATEGORY_MAPPING__
    if '__EQUIVALENT_CATEGORY_MAPPING__' not in globals():
        __EQUIVALENT_CATEGORY_MAPPING__ = defaultdict(set, util.load_or_create_cache('dbpedia_list_equivalents', _create_list_equivalents_mapping))
    return __EQUIVALENT_CATEGORY_MAPPING__[lst]


def _create_list_equivalents_mapping():
    util.get_logger().info('CACHE: Creating list-equivalents mapping')

    cat_graph = cat_base.get_merged_graph()
    all_categories_in_graph = cat_graph.get_all_categories()
    list_graph = list_base.get_merged_listgraph()

    # 1) find equivalent categories by exact name match
    name_to_cat_mapping = {cat_graph.get_name(node).replace('-', ' ').lower(): node for node in cat_graph.nodes}
    name_to_list_mapping = defaultdict(set)
    for lst in list_graph.nodes:
        name_to_list_mapping[list_util.list2name(lst).replace('-', ' ').lower()].add(lst)
    matching_names = set(name_to_list_mapping).intersection(set(name_to_cat_mapping))
    list_to_cat_exact_mapping = defaultdict(set, {lst: {name_to_cat_mapping[name]} for name in matching_names for lst in name_to_list_mapping[name]})
    util.get_logger().debug(f'Exact Mapping: Mapped {len(list_to_cat_exact_mapping)} lists to categories.')

    # 2) find equivalent categories by synonym match
    list_to_cat_synonym_mapping = defaultdict(set)

    cats_important_words = {cat: nlp_util.without_stopwords(nlp_util.get_canonical_name(cat_graph.get_name(cat), disable_normalization=False)) for cat in cat_graph.nodes}

    remaining_lists = list_graph.nodes.difference(set(list_to_cat_exact_mapping))
    for lst in remaining_lists:
        lst_important_words = nlp_util.without_stopwords(nlp_util.get_canonical_name(list_util.list2name(lst)))

        if list_util.is_listcategory(lst):
            candidates = cat_store.get_parents(lst)
        else:
            candidates = cat_store.get_topic_categories(lst) | cat_store.get_resource_categories(lst)
        candidates = {cand for cat in candidates for cand in cat_graph.get_nodes_for_category(cat) if cat in all_categories_in_graph}

        for candidate_cat in candidates:
            cat_important_words = cats_important_words[candidate_cat]
            if hypernymy_util.phrases_are_synonymous(lst_important_words, cat_important_words):
                list_to_cat_synonym_mapping[lst].add(candidate_cat)

    util.get_logger().debug(f'Synonym Mapping: Mapped {len(list_to_cat_synonym_mapping)} lists to {sum(len(cat) for cat in list_to_cat_synonym_mapping.values())} categories.')

    # merge mappings
    mapped_lsts = set(list_to_cat_exact_mapping) | set(list_to_cat_synonym_mapping)
    list_to_cat_equivalence_mapping = {lst: list_to_cat_exact_mapping[lst] | list_to_cat_synonym_mapping[lst] for lst in mapped_lsts}
    util.get_logger().debug(f'Equivalence Mapping: Mapped {len(list_to_cat_equivalence_mapping)} lists to {sum(len(cat) for cat in list_to_cat_equivalence_mapping.values())} categories.')
    return list_to_cat_equivalence_mapping


def get_parent_categories(lst: str) -> set:
    global __PARENT_CATEGORIES_MAPPING__
    if '__PARENT_CATEGORIES_MAPPING__' not in globals():
        __PARENT_CATEGORIES_MAPPING__ = defaultdict(set, util.load_or_create_cache('dbpedia_list_parents', _create_list_parents_mapping))
    return __PARENT_CATEGORIES_MAPPING__[lst]


def _create_list_parents_mapping():
    util.get_logger().info('CACHE: Creating list-parents mapping')

    cat_graph = cat_base.get_merged_graph()
    all_categories_in_graph = cat_graph.get_all_categories()
    list_graph = list_base.get_merged_listgraph()

    # 1) find parent categories by hypernym match
    list_to_cat_hypernym_mapping = defaultdict(set)

    cats_headlemmas = {cat: nlp_util.get_head_lemmas(cat_graph.get_name(cat)) for cat in cat_graph.nodes}

    unmapped_lists = {lst for lst in list_graph.nodes if not get_equivalent_categories(lst)}
    lsts_headlemmas = {lst: nlp_util.get_head_lemmas(list_util.list2name(lst), disable_normalization=True) for lst in unmapped_lists}
    for lst, lst_headlemmas in lsts_headlemmas.items():
        if list_util.is_listcategory(lst):
            candidates = cat_store.get_parents(lst)
        else:
            candidates = cat_store.get_topic_categories(lst) | cat_store.get_resource_categories(lst)
        candidates = {cand for cat in candidates for cand in cat_graph.get_nodes_for_category(cat) if cat in all_categories_in_graph}

        for candidate_cat in candidates:
            cat_headlemmas = cats_headlemmas[candidate_cat]
            if all(any(hypernymy_util.is_hypernym(chl, lhl) for chl in cat_headlemmas) for lhl in lst_headlemmas):
                list_to_cat_hypernym_mapping[lst].add(candidate_cat)
    util.get_logger().debug(f'Hypernym Mapping: Mapped {len(list_to_cat_hypernym_mapping)} lists to {sum(len(cat) for cat in list_to_cat_hypernym_mapping.values())} categories.')

    # 2) map listcategory to headlemma category if a mapping to the root category is the only alternative
    list_to_cat_headlemma_mapping = defaultdict(set)

    unmapped_lists = unmapped_lists.difference(set(list_to_cat_hypernym_mapping))
    unmapped_root_lists = {lst for lst in unmapped_lists if lst in list_graph.nodes and list_graph.depth(lst) == 1}
    for lst in unmapped_root_lists:
        lst_headlemmas = lsts_headlemmas[lst]
        mapped_categories = {cat_graph.get_node_by_name(inflection.pluralize(lhl).title()) for lhl in lst_headlemmas if cat_graph.get_node_by_name(inflection.pluralize(lhl).title())}
        if mapped_categories:
            list_to_cat_headlemma_mapping[lst] = mapped_categories
    util.get_logger().debug(f'Headlemma Mapping: Mapped {len(list_to_cat_headlemma_mapping)} lists to {sum(len(cat) for cat in list_to_cat_headlemma_mapping.values())} categories.')

    # merge mappings
    mapped_lsts = set(list_to_cat_hypernym_mapping) | set(list_to_cat_headlemma_mapping)
    list_to_cat_parents_mapping = {lst: list_to_cat_hypernym_mapping[lst] | list_to_cat_headlemma_mapping[lst] for lst in mapped_lsts}
    util.get_logger().debug(f'Parents Mapping: Mapped {len(list_to_cat_parents_mapping)} lists to {sum(len(cat) for cat in list_to_cat_parents_mapping.values())} categories.')
    return list_to_cat_parents_mapping
