from impl.caligraph.graph import CaLiGraph
import util


def get_base_graph() -> CaLiGraph:
    global __BASE_GRAPH__
    if '__BASE_GRAPH__' not in globals():
        initializer = lambda: CaLiGraph.build_graph().resolve_cycles()
        __BASE_GRAPH__ = util.load_or_create_cache('caligraph_base', initializer)
    return __BASE_GRAPH__
