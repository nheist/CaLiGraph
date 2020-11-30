"""Functionality to retrieve cached versions of caligraph in several stages."""

from impl.caligraph.graph import CaLiGraph
import impl.caligraph.serialize as cali_serialize
import impl.list.base as list_base
import util


def get_base_graph() -> CaLiGraph:
    """Retrieve graph created from categories and lists."""
    global __BASE_GRAPH__
    if '__BASE_GRAPH__' not in globals():
        initializer = lambda: CaLiGraph.build_graph().append_unconnected()
        __BASE_GRAPH__ = util.load_or_create_cache('caligraph_base', initializer)
    return __BASE_GRAPH__


def get_merged_ontology_graph() -> CaLiGraph:
    """Retrieve base graph joined with DBpedia ontology."""
    global __MERGED_ONTOLOGY_GRAPH__
    if '__MERGED_ONTOLOGY_GRAPH__' not in globals():
        initializer = lambda: get_base_graph().copy().merge_ontology(False).append_unconnected()
        __MERGED_ONTOLOGY_GRAPH__ = util.load_or_create_cache('caligraph_merged_ontology', initializer)
    return __MERGED_ONTOLOGY_GRAPH__


def get_filtered_graph() -> CaLiGraph:
    """Retrieve initial CaLiGraph (with resources extracted from list pages)."""
    global __FILTERED_GRAPH__
    if '__FILTERED_GRAPH__' not in globals():
        # first make sure that resources have already been extracted using the merged ontology graph
        list_base.get_listpage_entities(get_merged_ontology_graph(), '')

        initializer = lambda: get_base_graph().copy().merge_ontology(True).append_unconnected()
        __FILTERED_GRAPH__ = util.load_or_create_cache('caligraph_filtered', initializer)
    return __FILTERED_GRAPH__


def get_axiom_graph() -> CaLiGraph:
    """Retrieve CaLiGraph enriched with axioms from the Cat2Ax approach."""
    global __AXIOM_GRAPH__
    if '__AXIOM_GRAPH__' not in globals():
        initializer = lambda: get_filtered_graph().compute_axioms().remove_transitive_edges()
        __AXIOM_GRAPH__ = util.load_or_create_cache('caligraph_axiomatized', initializer)
    return __AXIOM_GRAPH__


def serialize_final_graph():
    """Serialize the final CaLiGraph."""
    graph = get_axiom_graph()
    cali_serialize.serialize_graph(graph)
