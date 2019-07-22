import networkx as nx
import util
from impl.util.base_graph import BaseGraph
import impl.util.rdf as rdf_util
import impl.category.base as cat_base
import impl.category.store as cat_store
import impl.category.util as cat_util
import impl.category.nlp as cat_nlp
import impl.dbpedia.store as dbp_store
from collections import defaultdict


class CaLiGraph(BaseGraph):

    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or rdf_util.CLASS_OWL_THING)

    @staticmethod
    def build_graph():
        # todo: initialise from category- and list-graph (by merging those two)
        pass
