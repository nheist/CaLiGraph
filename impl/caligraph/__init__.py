"""Functionality to retrieve cached versions of caligraph in several stages."""

from impl.caligraph.graph import CaLiGraph
import impl.caligraph.serialize as cali_serialize
from impl import listing
import utils


def get_base_graph() -> CaLiGraph:
    """Retrieve graph created from categories and lists."""
    global __BASE_GRAPH__
    if '__BASE_GRAPH__' not in globals():
        initializer = lambda: CaLiGraph.build_graph().append_unconnected()
        __BASE_GRAPH__ = utils.load_or_create_cache('caligraph_base', initializer)
    return __BASE_GRAPH__


def get_merged_ontology_graph() -> CaLiGraph:
    """Retrieve base graph joined with DBpedia ontology."""
    global __MERGED_ONTOLOGY_GRAPH__
    if '__MERGED_ONTOLOGY_GRAPH__' not in globals():
        initializer = lambda: get_base_graph().copy().merge_ontology(False).append_unconnected()
        __MERGED_ONTOLOGY_GRAPH__ = utils.load_or_create_cache('caligraph_merged_ontology', initializer)
    return __MERGED_ONTOLOGY_GRAPH__


def get_axiom_graph() -> CaLiGraph:
    """Retrieve CaLiGraph enriched with axioms from the Cat2Ax approach."""
    global __AXIOM_GRAPH__
    if '__AXIOM_GRAPH__' not in globals():
        initializer = lambda: get_merged_ontology_graph().remove_transitive_edges().compute_axioms()
        __AXIOM_GRAPH__ = utils.load_or_create_cache('caligraph_axiomatized', initializer)
    return __AXIOM_GRAPH__


def serialize_final_graph():
    """Serialize the final CaLiGraph."""
    graph = get_axiom_graph()

    listing.get_page_entities(graph)  # extract entities from listings all over Wikipedia
    graph.enable_listing_resources()  # finally, enable use of the extracted listing resources in the graph

    cali_serialize.serialize_graph(graph)
