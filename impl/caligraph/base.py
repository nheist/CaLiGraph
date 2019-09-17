from impl.caligraph.graph import CaLiGraph
import util


def get_base_graph() -> CaLiGraph:
    global __BASE_GRAPH__
    if '__BASE_GRAPH__' not in globals():
        initializer = lambda: CaLiGraph.build_graph().resolve_cycles()
        __BASE_GRAPH__ = util.load_or_create_cache('caligraph_base', initializer)
    return __BASE_GRAPH__


# TODO: check for cycles
def get_merged_ontology_graph() -> CaLiGraph:
    global __MERGED_ONTOLOGY_GRAPH__
    if '__MERGED_ONTOLOGY_GRAPH__' not in globals():
        initializer = lambda: get_base_graph().merge_ontology(False)
        __MERGED_ONTOLOGY_GRAPH__ = util.load_or_create_cache('caligraph_merged_ontology', initializer)
    return __MERGED_ONTOLOGY_GRAPH__


# TODO: check for cycles
def get_filtered_graph() -> CaLiGraph:
    global __FILTERED_GRAPH__
    if '__FILTERED_GRAPH__' not in globals():
        initializer = lambda: get_merged_ontology_graph().merge_ontology(True)
        __FILTERED_GRAPH__ = util.load_or_create_cache('caligraph_filtered', initializer)
    return __FILTERED_GRAPH__

# TODO: check whether we are generating any disjoint type mappings
