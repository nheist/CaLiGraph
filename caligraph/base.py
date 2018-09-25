from caligraph.category.graph import CategoryGraph
import util


def get_conceptual_category_graph() -> CategoryGraph:
    global __CONCEPTUAL_CATGRAPH__
    if '__CONCEPTUAL_CATGRAPH__' not in globals():
        initializer = lambda: CategoryGraph.create_from_dbpedia().remove_unconnected().make_conceptual()
        __CONCEPTUAL_CATGRAPH__ = util.load_or_create_cache('catgraph_conceptual', initializer)

    return __CONCEPTUAL_CATGRAPH__


def get_cycle_free_category_graph() -> CategoryGraph:
    global __CYCLEFREE_CATGRAPH__
    if '__CYCLEFREE_CATGRAPH__' not in globals():
        initializer = lambda: get_conceptual_category_graph().resolve_cycles()
        __CYCLEFREE_CATGRAPH__ = util.load_or_create_cache('catgraph_cyclefree', initializer)

    return __CYCLEFREE_CATGRAPH__


def get_dbpedia_typed_category_graph() -> CategoryGraph:
    pass
