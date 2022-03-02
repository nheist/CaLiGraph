"""Functionality to retrieve cached versions of the category graph in several stages."""

from .graph import CategoryGraph
import utils


# TODO: analyse methods from papers to probably improve the hierarchy:
#  - 'Derivation of “is a” taxonomy from Wikipedia Category Graph'
#  - 'Revisiting Taxonomy Induction over Wikipedia'


def get_conceptual_category_graph() -> CategoryGraph:
    """Retrieve category graph with filtered categories."""
    global __CONCEPTUAL_CATEGORY_GRAPH__
    if '__CONCEPTUAL_CATEGORY_GRAPH__' not in globals():
        __CONCEPTUAL_CATEGORY_GRAPH__ = utils.load_or_create_cache('catgraph_conceptual', CategoryGraph.create_from_dbpedia)
    return __CONCEPTUAL_CATEGORY_GRAPH__


def get_wikitaxonomy_graph() -> CategoryGraph:
    """Retrieve category graph with filtered categories and edges."""
    global __WIKITAXONOMY_CATEGORY_GRAPH__
    if '__WIKITAXONOMY_CATEGORY_GRAPH__' not in globals():
        initializer = lambda: get_conceptual_category_graph().remove_unrelated_edges()
        __WIKITAXONOMY_CATEGORY_GRAPH__ = utils.load_or_create_cache('catgraph_wikitaxonomy', initializer)
    return __WIKITAXONOMY_CATEGORY_GRAPH__


def get_merged_graph() -> CategoryGraph:
    """Retrieve the cycle-free category graph with filtered+merged categories and filtered edges."""
    global __MERGED_GRAPH__
    if '__MERGED_GRAPH__' not in globals():
        initializer = lambda: get_wikitaxonomy_graph().merge_nodes().remove_transitive_edges()
        __MERGED_GRAPH__ = utils.load_or_create_cache('catgraph_merged', initializer)
    return __MERGED_GRAPH__
