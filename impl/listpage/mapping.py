"""Mapping of listpages to the category-hierarchy

A mapping can have one of two types:
- Equivalence Mapping: The entities of the listpage are put directly into the mapped hierarchy item, if:
  - the name of the list is equivalent to the name of a category
  - every word in the list's name is synonymous to a word in the category's name (and vice versa) [reduce complexity?]
- Child Mapping: The listpage is appended to the hierarchy as a child of an existing hierarchy item, if:
  - the headlemmas of a parent category are synonyms/hypernyms of the headlemmas of the list
  - the only parent of the list is root -> append it to root
"""

from typing import Set, Dict
from impl import category
from impl import listpage
import utils
from utils import get_logger
import impl.util.nlp as nlp_util
import impl.util.hypernymy as hypernymy_util
from collections import defaultdict
from impl.dbpedia.resource import DbpResourceStore
from impl.dbpedia.category import DbpCategoryStore


def get_related_category_nodes(list_node: str) -> Set[str]:
    return get_equivalent_category_nodes(list_node) | get_parent_category_nodes(list_node)


def get_equivalent_category_nodes(list_node: str) -> Set[str]:
    global __EQUIVALENT_CATEGORY_MAPPING__
    if '__EQUIVALENT_CATEGORY_MAPPING__' not in globals():
        __EQUIVALENT_CATEGORY_MAPPING__ = defaultdict(set, utils.load_or_create_cache('dbpedia_list_equivalents', _create_list_equivalents_mapping))
    return __EQUIVALENT_CATEGORY_MAPPING__[list_node]


def _create_list_equivalents_mapping() -> Dict[str, Set[str]]:
    get_logger().debug('Creating list-equivalents mapping..')

    cat_graph = category.get_merged_graph()
    list_graph = listpage.get_merged_listgraph()

    # 1) find equivalent categories by exact name match
    label_to_cat_mapping = {cat_graph.get_label(node).replace('-', ' ').lower(): node for node in cat_graph.nodes}
    label_to_list_mapping = defaultdict(set)
    for list_node in list_graph.content_nodes:
        label_to_list_mapping[list_graph.get_label(list_node).replace('-', ' ').lower()].add(list_node)
    matching_labels = set(label_to_list_mapping).intersection(set(label_to_cat_mapping))
    list_to_cat_exact_mapping = defaultdict(set, {list_node: {label_to_cat_mapping[label]} for label in matching_labels for list_node in label_to_list_mapping[label]})
    get_logger().debug(f'Mapped {len(list_to_cat_exact_mapping)} lists to categories with exact mapping.')

    # 2) find equivalent categories by synonym match
    list_to_cat_synonym_mapping = defaultdict(set)
    cats_important_words = {cat: nlp_util.without_stopwords(nlp_util.get_canonical_label(cat_graph.get_label(cat))) for cat in cat_graph.nodes}
    remaining_lists = list_graph.content_nodes.difference(set(list_to_cat_exact_mapping))
    for list_node in remaining_lists:
        lst_important_words = nlp_util.without_stopwords(nlp_util.get_canonical_label(list_graph.get_label(list_node)))
        candidate_cats = _get_candidate_category_nodes_for_list_node(list_node, cat_graph)

        exact_matches = {c for c in candidate_cats if hypernymy_util.phrases_are_equal(lst_important_words, cats_important_words[c])}
        if exact_matches:
            list_to_cat_synonym_mapping[list_node].update(exact_matches)
            continue
        synonym_matches = {c for c in candidate_cats if hypernymy_util.phrases_are_synonymous(lst_important_words, cats_important_words[c])}
        if len(synonym_matches) == 1:  # if we have more than one synonym match, we ignore them as they are not specific enough
            list_to_cat_synonym_mapping[list_node].update(synonym_matches)
    get_logger().debug(f'Mapped {len(list_to_cat_synonym_mapping)} lists to {sum(len(cat) for cat in list_to_cat_synonym_mapping.values())} categories with synonym mapping.')

    # merge mappings
    mapped_lists = set(list_to_cat_exact_mapping) | set(list_to_cat_synonym_mapping)
    list_to_cat_equivalence_mapping = {list_node: list_to_cat_exact_mapping[list_node] | list_to_cat_synonym_mapping[list_node] for list_node in mapped_lists}
    get_logger().debug(f'Mapped {len(list_to_cat_equivalence_mapping)} lists to {sum(len(cat) for cat in list_to_cat_equivalence_mapping.values())} categories with equivalence mapping.')
    return list_to_cat_equivalence_mapping


def get_parent_category_nodes(list_node: str) -> Set[str]:
    global __PARENT_CATEGORIES_MAPPING__
    if '__PARENT_CATEGORIES_MAPPING__' not in globals():
        __PARENT_CATEGORIES_MAPPING__ = defaultdict(set, utils.load_or_create_cache('dbpedia_list_parents', _create_list_parents_mapping))
    return __PARENT_CATEGORIES_MAPPING__[list_node]


def _create_list_parents_mapping() -> Dict[str, Set[str]]:
    get_logger().debug('Creating list-parents mapping..')

    cat_graph = category.get_merged_graph()
    cats_LHS = cat_graph.get_node_LHS()
    list_graph = listpage.get_merged_listgraph()
    lists_LHS = list_graph.get_node_LHS()

    # 1) find parent categories by hypernym match
    list_to_cat_hypernym_mapping = defaultdict(set)
    unmapped_lists = {list_node for list_node in list_graph.content_nodes if not get_equivalent_category_nodes(list_node)}
    for list_node in unmapped_lists:
        for candidate_cat_node in _get_candidate_category_nodes_for_list_node(list_node, cat_graph):
            cat_subjectlemmas = cats_LHS[candidate_cat_node]
            lst_subjectlemmas = lists_LHS[list_node]
            if all(any(hypernymy_util.is_hypernym(chl, lhl) for chl in cat_subjectlemmas) for lhl in lst_subjectlemmas):
                list_to_cat_hypernym_mapping[list_node].add(candidate_cat_node)
    get_logger().debug(f'Mapped {len(list_to_cat_hypernym_mapping)} lists to {sum(len(cat) for cat in list_to_cat_hypernym_mapping.values())} categories with hypernym mapping.')

    # 2) map listcategory to headlemma category as a mapping to the root category is the only alternative
    unmapped_lists = unmapped_lists.difference(set(list_to_cat_hypernym_mapping))
    unmapped_head_lists = {list_node for list_node in unmapped_lists if list_graph.root_node in list_graph.parents(list_node)}
    list_to_cat_headlemma_mapping = defaultdict(set, cat_graph.find_parents_by_headlemma_match(unmapped_head_lists, list_graph))
    get_logger().debug(f'Mapped {len(list_to_cat_headlemma_mapping)} lists to {sum(len(cat) for cat in list_to_cat_headlemma_mapping.values())} categories with headlemma mapping.')

    # merge mappings
    mapped_lsts = set(list_to_cat_hypernym_mapping) | set(list_to_cat_headlemma_mapping)
    list_to_cat_parents_mapping = {list_node: list_to_cat_hypernym_mapping[list_node] | list_to_cat_headlemma_mapping[list_node] for list_node in mapped_lsts}
    get_logger().debug(f'Mapped {len(list_to_cat_parents_mapping)} lists to {sum(len(cat) for cat in list_to_cat_parents_mapping.values())} categories with parents mapping.')
    return list_to_cat_parents_mapping


def _get_candidate_category_nodes_for_list_node(list_node: str, cat_graph) -> Set[str]:
    dbc = DbpCategoryStore.instance()
    dbr = DbpResourceStore.instance()
    candidates = set()
    if dbc.has_category_with_name(list_node):  # listcategory
        listcat = dbc.get_category_by_name(list_node)
        candidates.update(dbc.get_parents(listcat))
    if dbr.has_resource_with_name(list_node):  # listpage
        lp = dbr.get_resource_by_name(list_node)
        candidates.update(dbc.get_categories_for_topic(lp) | dbc.get_categories_for_resource(lp))
    return {n for cat in candidates for n in cat_graph.get_nodes_for_category(cat)}
