from .graph import CategoryGraph
import util


def get_conceptual_category_graph() -> CategoryGraph:
    global __CONCEPTUAL_CATEGORY_GRAPH__
    if '__CONCEPTUAL_CATEGORY_GRAPH__' not in globals():
        initializer = lambda: CategoryGraph.create_from_dbpedia().remove_unconnected().make_conceptual()
        __CONCEPTUAL_CATEGORY_GRAPH__ = util.load_or_create_cache('catgraph_conceptual', initializer)
    return __CONCEPTUAL_CATEGORY_GRAPH__


def get_wikitaxonomy_graph() -> CategoryGraph:
    global __WIKITAXONOMY_CATEGORY_GRAPH__
    if '__WIKITAXONOMY_CATEGORY_GRAPH__' not in globals():
        initializer = lambda: get_conceptual_category_graph().remove_unrelated_edges().append_unconnected().remove_unconnected()
        __WIKITAXONOMY_CATEGORY_GRAPH__ = util.load_or_create_cache('catgraph_wikitaxonomy', initializer)
    return __WIKITAXONOMY_CATEGORY_GRAPH__


def get_cyclefree_wikitaxonomy_graph() -> CategoryGraph:
    global __CYCLEFREE_WIKITAXONOMY_GRAPH__
    if '__CYCLEFREE_WIKITAXONOMY_GRAPH__' not in globals():
        initializer = lambda: get_wikitaxonomy_graph().resolve_cycles()
        __CYCLEFREE_WIKITAXONOMY_GRAPH__ = util.load_or_create_cache('catgraph_cyclefree', initializer)
    return __CYCLEFREE_WIKITAXONOMY_GRAPH__


def get_merged_graph() -> CategoryGraph:
    global __MERGED_GRAPH__
    if '__MERGED_GRAPH__' not in globals():
        initializer = lambda: get_cyclefree_wikitaxonomy_graph().merge_nodes()
        __MERGED_GRAPH__ = util.load_or_create_cache('catgraph_merged', initializer)
    return __MERGED_GRAPH__