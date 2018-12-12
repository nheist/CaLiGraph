from .graph import CategoryGraph
import util


def get_conceptual_category_graph() -> CategoryGraph:
    global __CONCEPTUAL_CATEGORY_GRAPH__
    if '__CONCEPTUAL_CATEGORY_GRAPH__' not in globals():
        initializer = lambda: CategoryGraph.create_from_dbpedia().remove_unconnected().make_conceptual()
        __CONCEPTUAL_CATEGORY_GRAPH__ = util.load_or_create_cache('catgraph_conceptual', initializer)
    return __CONCEPTUAL_CATEGORY_GRAPH__


def get_cycle_free_category_graph() -> CategoryGraph:
    global __CYCLEFREE_CATEGORY_GRAPH__
    if '__CYCLEFREE_CATEGORY_GRAPH__' not in globals():
        initializer = get_conceptual_category_graph().resolve_cycles()
        __CYCLEFREE_CATEGORY_GRAPH__ = util.load_or_create_cache('catgraph_cyclefree', initializer)
    return __CYCLEFREE_CATEGORY_GRAPH__


def get_dbp_typed_category_graph() -> CategoryGraph:
    global __DBPTYPED_CATEGORY_GRAPH__
    if '__DBPTYPED_CATEGORY_GRAPH__' not in globals():
        initializer = lambda: get_cycle_free_category_graph().assign_dbp_types()
        __DBPTYPED_CATEGORY_GRAPH__ = util.load_or_create_cache('catgraph_dbp_typed', initializer)
    return __DBPTYPED_CATEGORY_GRAPH__


def get_taxonomic_category_graph() -> CategoryGraph:
    global __TAXONOMIC_CATEGORY_GRAPH__
    if '__TAXONOMIC_CATEGORY_GRAPH__' not in globals():
        initializer = lambda: get_dbp_typed_category_graph().make_taxonomy()
        __TAXONOMIC_CATEGORY_GRAPH__ = util.load_or_create_cache('catgraph_taxonomic', initializer)
    return __TAXONOMIC_CATEGORY_GRAPH__
