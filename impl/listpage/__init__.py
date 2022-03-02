"""Functionality to retrieve cached versions of the list graph in several stages."""

import utils
from impl.listpage.graph import ListGraph


# LIST HIERARCHY

def get_base_listgraph() -> ListGraph:
    """Retrieve basic list graph without any modifications."""
    global __BASE_LISTGRAPH__
    if '__BASE_LISTGRAPH__' not in globals():
        __BASE_LISTGRAPH__ = utils.load_or_create_cache('listgraph_base', ListGraph.create_from_dbpedia)
    return __BASE_LISTGRAPH__


def get_wikitaxonomy_listgraph() -> ListGraph:
    """Retrieve list graph with filtered edges."""
    global __WIKITAXONOMY_LISTGRAPH__
    if '__WIKITAXONOMY_LISTGRAPH__' not in globals():
        initializer = lambda: get_base_listgraph().remove_unrelated_edges()
        __WIKITAXONOMY_LISTGRAPH__ = utils.load_or_create_cache('listgraph_wikitaxonomy', initializer)
    return __WIKITAXONOMY_LISTGRAPH__


def get_merged_listgraph() -> ListGraph:
    """Retrieve list graph with filtered edges, resolved cycles, and merged lists."""
    global __MERGED_LISTGRAPH__
    if '__MERGED_LISTGRAPH__' not in globals():
        initializer = lambda: get_wikitaxonomy_listgraph().merge_nodes().remove_leaf_listcategories().remove_transitive_edges()
        __MERGED_LISTGRAPH__ = utils.load_or_create_cache('listgraph_merged', initializer)
    return __MERGED_LISTGRAPH__

