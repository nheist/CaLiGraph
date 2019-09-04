from impl.caligraph.graph import CaLiGraph
import util


def get_base_graph() -> CaLiGraph:
    global __BASE_GRAPH__
    if '__BASE_GRAPH__' not in globals():
        __BASE_GRAPH__ = util.load_or_create_cache('caligraph_base', CaLiGraph.build_graph)
    return __BASE_GRAPH__
