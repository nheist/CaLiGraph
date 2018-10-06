from .graph import CategoryGraph
import util


def get_conceptual_category_graph() -> CategoryGraph:
    initializer = lambda: CategoryGraph.create_from_dbpedia().remove_unconnected().make_conceptual()
    return util.load_or_create_cache('catgraph_conceptual', initializer)


def get_cycle_free_category_graph() -> CategoryGraph:
    initializer = lambda: get_conceptual_category_graph().resolve_cycles()
    return util.load_or_create_cache('catgraph_cyclefree', initializer)


def get_dbp_typed_category_graph() -> CategoryGraph:
    initializer = lambda: get_cycle_free_category_graph().assign_dbp_types()
    return util.load_or_create_cache('catgraph_dbp_typed', initializer)
