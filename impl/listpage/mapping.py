"""Mapping of listpages to the category-hierarchy

A mapping can have one of two types:
- Equivalence Mapping: The entities of the listpage are put directly into the mapped hierarchy item, if:
  - the name of the list is equivalent to the name of a category
  - every word in the list's name is synonymous to a word in the category's name (and vice versa) [reduce complexity?]
- Child Mapping: The listpage is appended to the hierarchy as a child of an existing hierarchy item, if:
  - the headlemmas of a parent category are synonyms/hypernyms of the headlemmas of the list
  - the only parent of the list is root -> append it to root
"""

from impl import category
import impl.category.store as cat_store
import impl.listpage.util as list_util
from impl import listpage
import utils
from utils import get_logger
import impl.util.nlp as nlp_util
import impl.util.hypernymy as hypernymy_util
from collections import defaultdict


def get_equivalent_categories(lst: str) -> set:
    global __EQUIVALENT_CATEGORY_MAPPING__
    if '__EQUIVALENT_CATEGORY_MAPPING__' not in globals():
        __EQUIVALENT_CATEGORY_MAPPING__ = defaultdict(set, utils.load_or_create_cache('dbpedia_list_equivalents', _create_list_equivalents_mapping))
    return __EQUIVALENT_CATEGORY_MAPPING__[lst]


def _create_list_equivalents_mapping():
    get_logger().debug('Creating list-equivalents mapping..')

    cat_graph = category.get_merged_graph()
    list_graph = listpage.get_merged_listgraph()

    # 1) find equivalent categories by exact name match
    name_to_cat_mapping = {cat_graph.get_name(node).replace('-', ' ').lower(): node for node in cat_graph.nodes}
    name_to_list_mapping = defaultdict(set)
    for lst in list_graph.content_nodes:
        name_to_list_mapping[list_util.list2name(lst).replace('-', ' ').lower()].add(lst)
    matching_names = set(name_to_list_mapping).intersection(set(name_to_cat_mapping))
    list_to_cat_exact_mapping = defaultdict(set, {lst: {name_to_cat_mapping[name]} for name in matching_names for lst in name_to_list_mapping[name]})
    get_logger().debug(f'Mapped {len(list_to_cat_exact_mapping)} lists to categories with exact mapping.')

    # 2) find equivalent categories by synonym match
    list_to_cat_synonym_mapping = defaultdict(set)

    cats_important_words = {cat: nlp_util.without_stopwords(nlp_util.get_canonical_name(cat_graph.get_name(cat))) for cat in cat_graph.nodes}
    remaining_lists = list_graph.content_nodes.difference(set(list_to_cat_exact_mapping))
    for lst in remaining_lists:
        lst_important_words = nlp_util.without_stopwords(nlp_util.get_canonical_name(list_util.list2name(lst)))
        for candidate_cat in _get_candidate_categories_for_list(lst, cat_graph):
            cat_important_words = cats_important_words[candidate_cat]
            if hypernymy_util.phrases_are_synonymous(lst_important_words, cat_important_words):
                list_to_cat_synonym_mapping[lst].add(candidate_cat)

    get_logger().debug(f'Mapped {len(list_to_cat_synonym_mapping)} lists to {sum(len(cat) for cat in list_to_cat_synonym_mapping.values())} categories with synonym mapping.')

    # merge mappings
    mapped_lsts = set(list_to_cat_exact_mapping) | set(list_to_cat_synonym_mapping)
    list_to_cat_equivalence_mapping = {lst: list_to_cat_exact_mapping[lst] | list_to_cat_synonym_mapping[lst] for lst in mapped_lsts}
    get_logger().debug(f'Mapped {len(list_to_cat_equivalence_mapping)} lists to {sum(len(cat) for cat in list_to_cat_equivalence_mapping.values())} categories with equivalence mapping.')
    return list_to_cat_equivalence_mapping


def get_parent_categories(lst: str) -> set:
    global __PARENT_CATEGORIES_MAPPING__
    if '__PARENT_CATEGORIES_MAPPING__' not in globals():
        __PARENT_CATEGORIES_MAPPING__ = defaultdict(set, utils.load_or_create_cache('dbpedia_list_parents', _create_list_parents_mapping))
    return __PARENT_CATEGORIES_MAPPING__[lst]


def _create_list_parents_mapping():
    get_logger().debug('Creating list-parents mapping..')

    cat_graph = category.get_merged_graph()
    cats_LHS = cat_graph.get_node_LHS()
    list_graph = listpage.get_merged_listgraph()
    lists_LHS = list_graph.get_node_LHS()

    # 1) find parent categories by hypernym match
    list_to_cat_hypernym_mapping = defaultdict(set)
    unmapped_lists = {lst for lst in list_graph.content_nodes if not get_equivalent_categories(lst)}
    for lst in unmapped_lists:
        for candidate_cat in _get_candidate_categories_for_list(lst, cat_graph):
            cat_subjectlemmas = cats_LHS[candidate_cat]
            lst_subjectlemmas = lists_LHS[lst]
            if all(any(hypernymy_util.is_hypernym(chl, lhl) for chl in cat_subjectlemmas) for lhl in lst_subjectlemmas):
                list_to_cat_hypernym_mapping[lst].add(candidate_cat)
    get_logger().debug(f'Mapped {len(list_to_cat_hypernym_mapping)} lists to {sum(len(cat) for cat in list_to_cat_hypernym_mapping.values())} categories with hypernym mapping.')

    # 2) map listcategory to headlemma category as a mapping to the root category is the only alternative
    unmapped_lists = unmapped_lists.difference(set(list_to_cat_hypernym_mapping))
    unmapped_head_lists = {lst for lst in unmapped_lists if list_graph.root_node in list_graph.parents(lst)}

    list_to_cat_headlemma_mapping = defaultdict(set, cat_graph.find_parents_by_headlemma_match(unmapped_head_lists, list_graph))
    get_logger().debug(f'Mapped {len(list_to_cat_headlemma_mapping)} lists to {sum(len(cat) for cat in list_to_cat_headlemma_mapping.values())} categories with headlemma mapping.')

    # merge mappings
    mapped_lsts = set(list_to_cat_hypernym_mapping) | set(list_to_cat_headlemma_mapping)
    list_to_cat_parents_mapping = {lst: list_to_cat_hypernym_mapping[lst] | list_to_cat_headlemma_mapping[lst] for lst in mapped_lsts}
    get_logger().debug(f'Mapped {len(list_to_cat_parents_mapping)} lists to {sum(len(cat) for cat in list_to_cat_parents_mapping.values())} categories with parents mapping.')
    return list_to_cat_parents_mapping


def _get_candidate_categories_for_list(lst, cat_graph) -> set:
    if list_util.is_listcategory(lst):  # list category
        candidates = cat_store.get_parents(lst)
    else:  # list page
        candidates = cat_store.get_topic_categories(lst) | cat_store.get_resource_categories(lst)
    return {n for cat in candidates for n in cat_graph.get_nodes_for_category(cat)}
