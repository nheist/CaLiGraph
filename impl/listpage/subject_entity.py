"""Heuristic retrieval of subject entities in list pages using the mapping to DBpedia categories."""

import impl.dbpedia.util as dbp_util
import impl.dbpedia.heuristics as dbp_heur
import impl.dbpedia.store as dbp_store
import impl.category.store as cat_store
from impl import category
import impl.listpage.mapping as list_mapping


def find_subject_entities_for_listpage(page_uri: str, page_data: dict, graph) -> tuple:
    """Retrieve all entities of a list page that are subject entities or explicit non-subject entities."""
    positive_SEs, negative_SEs = set(), set()
    # compute potential subject entities for list page
    page_potential_SEs = {dbp_util.resource2name(res) for cat in _get_category_descendants_for_list(page_uri) for res in cat_store.get_resources(cat)}
    # compute types of list page
    page_types = {t for n in graph.get_nodes_for_part(page_uri) for t in dbp_store.get_independent_types(graph.get_transitive_dbpedia_type_closure(n))}
    page_disjoint_types = {dt for t in page_types for dt in dbp_heur.get_all_disjoint_types(t)}
    # collect all linked entities on the page
    page_entities = {ent['name'] for s in page_data['sections'] for enum in s['enums'] for entry in enum for ent in entry['entities']}
    page_entities.update({ent['name'] for s in page_data['sections'] for table in s['tables'] for row in table['data'] for cell in row for ent in cell['entities']})
    for ent in page_entities:
        ent_uri = dbp_util.name2resource(ent)
        if not dbp_store.is_possible_resource(ent_uri):
            negative_SEs.add(ent)
        elif ent in page_potential_SEs:
            positive_SEs.add(ent)
        elif page_disjoint_types.intersection(dbp_store.get_types(ent_uri)):
            negative_SEs.add(ent)
    return positive_SEs, negative_SEs


def _get_category_descendants_for_list(listpage_uri: str) -> set:
    """Return the category that is most closely related to the given list page as well as all of its children."""
    categories = set()
    cat_graph = category.get_merged_graph()
    mapped_categories = {x for cat in _get_categories_for_list(listpage_uri) for x in cat_graph.get_nodes_for_category(cat)}
    descendant_categories = {descendant for cat in mapped_categories for descendant in cat_graph.descendants(cat)}
    for cat in mapped_categories | descendant_categories:
        categories.update(cat_graph.get_categories(cat))
    return categories


def _get_categories_for_list(listpage_uri: str) -> set:
    """Return category that is mapped to the list page."""
    return list_mapping.get_equivalent_categories(listpage_uri) | list_mapping.get_parent_categories(listpage_uri)
