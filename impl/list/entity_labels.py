"""Extraction of features and labels for different list page layouts."""

import impl.list.mapping as list_mapping
import impl.dbpedia.store as dbp_store
import impl.dbpedia.heuristics as dbp_heur
import impl.category.store as cat_store
import impl.category.base as cat_base
import pandas as pd
import util
from collections import defaultdict


def assign_entity_labels(graph, df: pd.DataFrame):
    """Derive labels (1 for True, 0 for False, -1 for Undefined) for all entities in the data frame."""
    listpage_valid_resources = {}
    listpage_types = defaultdict(set)

    for idx, (listpage_uri, df_group) in enumerate(df.groupby('_page_uri')):
        if idx % 1000 == 0:
            util.get_logger().debug(f'LIST/FEATURES: Processed {idx} list pages.')
        listpage_resources = set(df_group['_entity_uri'].unique())
        listpage_category_resources = {res for cat in _get_category_descendants_for_list(listpage_uri) for res in cat_store.get_resources(cat)}
        listpage_valid_resources[listpage_uri] = listpage_resources.intersection(listpage_category_resources)

        caligraph_nodes = graph.get_nodes_for_part(listpage_uri)
        for n in caligraph_nodes:
            listpage_types[listpage_uri].update(dbp_store.get_independent_types(graph.get_dbpedia_types(n)))

    # assign positive and negative labels that are induced directly from the taxonomy
    df['label'] = df.apply(lambda row: _compute_label_for_entity(row['_page_uri'], row['_entity_uri'], row['_link_type'], listpage_valid_resources, listpage_types), axis=1)

    if util.get_config('page.extraction.use_negative_evidence_assumption'):
        # ASSUMPTION: if an entry has at least one positive example, all the unknown examples are negative
        # locate all entries that have a positive example
        lines_with_positive_example = set()
        for _, row in df[df['label'] == 1].iterrows():
            lines_with_positive_example.add(_get_listpage_line_id(row))
        # make all candidate examples negative that appear in an entry with a positive example
        for i, row in df[df['label'] == -1].iterrows():
            if _get_listpage_line_id(row) in lines_with_positive_example:
                df.at[i, 'label'] = 0


def _get_listpage_line_id(row: pd.Series) -> tuple:
    return row['_page_uri'], row['_line_idx']


def _get_sortkey(row: pd.Series):
    return row['_entity_line_idx']


def _compute_label_for_entity(listpage_uri: str, entity_uri: str, link_type: str, lp_valid_resources: dict, lp_types: dict) -> int:
    """Return a label for the entity based on links in the taxonomy graph."""
    if link_type != 'blue':
        return -1

    if not dbp_store.is_possible_resource(entity_uri):
        return 0
    if entity_uri in lp_valid_resources[listpage_uri]:
        return 1
    entity_types = dbp_store.get_types(entity_uri)
    if any(entity_types.intersection(dbp_heur.get_disjoint_types(t)) for t in lp_types[listpage_uri]):
        return 0
    return -1


def _get_category_descendants_for_list(listpage_uri: str) -> set:
    """Return the category that is most closely related to the given list page as well as all of its children."""
    categories = set()
    cat_graph = cat_base.get_merged_graph()
    mapped_categories = {x for cat in _get_categories_for_list(listpage_uri) for x in cat_graph.get_nodes_for_category(cat)}
    descendant_categories = {descendant for cat in mapped_categories for descendant in cat_graph.descendants(cat)}
    for cat in mapped_categories | descendant_categories:
        categories.update(cat_graph.get_categories(cat))
    return categories


def _get_categories_for_list(listpage_uri: str) -> set:
    """Return category that is mapped to the list page."""
    return list_mapping.get_equivalent_categories(listpage_uri) | list_mapping.get_parent_categories(listpage_uri)