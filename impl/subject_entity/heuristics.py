"""Heuristic retrieval of subject entities in list pages using the mapping to DBpedia categories."""

from typing import Tuple, Set
import impl.dbpedia.heuristics as dbp_heur
from impl import category
from impl import listpage
import impl.listpage.mapping as list_mapping
from impl.dbpedia.ontology import DbpOntologyStore
from impl.dbpedia.resource import DbpEntity, DbpListpage
from impl.dbpedia.category import DbpCategory
from impl.caligraph.ontology import ClgOntologyStore
from impl.wikipedia import WikipediaPage


def find_subject_entities_for_listpage(wp: WikipediaPage) -> Tuple[Set[int], Set[int]]:
    """Retrieve all entities of a list page that are subject entities or explicit non-subject entities."""
    clgo = ClgOntologyStore.instance()
    dbo = DbpOntologyStore.instance()
    # compute potential subject entities for list page
    subject_entity_candidates = {ent for cat in _get_category_descendants_for_list(wp.resource) for ent in cat.get_entities()}
    # find dbp types of listpage based on the caligraph ontology
    page_dbp_types = {dt for ct in clgo.get_types_for_associated_dbp_resource(wp.resource) for dt in dbo.get_independent_types(ct.get_all_dbp_types())}
    page_disjoint_types = {dt for t in page_dbp_types for dt in dbp_heur.get_all_disjoint_types(t)}

    subject_entities, non_subject_entities = set(), set()
    for ent in wp.get_listing_entities():
        if not isinstance(ent, DbpEntity):
            non_subject_entities.add(ent.idx)
        elif ent in subject_entity_candidates:
            subject_entities.add(ent.idx)
        elif page_disjoint_types.intersection(ent.get_types()):
            non_subject_entities.add(ent.idx)
    return subject_entities, non_subject_entities


def _get_category_descendants_for_list(lp: DbpListpage) -> Set[DbpCategory]:
    """Return the categories that are most closely related to the given listpage as well as all of its children."""
    list_graph = listpage.get_merged_listgraph()
    cat_graph = category.get_merged_graph()
    category_nodes = {cat_node for list_node in list_graph.get_nodes_for_list(lp) for cat_node in list_mapping.get_related_category_nodes(list_node)}
    category_nodes = category_nodes | {dn for cn in category_nodes for dn in cat_graph.descendants(cn)}

    categories = {cat for cn in category_nodes for cat in cat_graph.get_categories(cn)}
    return categories
