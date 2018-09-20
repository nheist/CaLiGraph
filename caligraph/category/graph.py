import networkx as nx
from . import store as cat_store
from . import nlp as cat_nlp
import caligraph.util.nlp as nlp_util
import util


class CategoryGraph:
    def __init__(self, graph: nx.DiGraph, root_node: str):
        self.graph = graph
        self.root_node = root_node

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def get_connected_graph(self):
        return CategoryGraph(nx.bfs_tree(self.graph, self.root_node), self.root_node)

    def get_conceptual_graph(self):
        categories = set(self.graph.nodes)
        # filtering maintenance categories
        categories = categories.difference(cat_store.get_maintenance_cats())
        # filtering administrative categories
        categories = [cat for cat in categories if not cat.endswith(('templates', 'navigational boxes'))]
        # filtering non-conceptual categories
        categories = [cat for cat in categories if cat_nlp.is_conceptual(cat) or cat == self.root_node]

        # persisting spacy cache so that parsed categories are cached
        nlp_util.persist_cache()

        # todo: connect unconnected nodes
        return CategoryGraph(self.graph.subgraph(categories), self.root_node)

    @classmethod
    def create_from_dbpedia(cls, root_node=None):
        edges = [(node, child) for node in cat_store.get_all_cats() for child in cat_store.get_children(node)]
        root_node = root_node if root_node else util.get_config('caligraph.category.root_node')
        return CategoryGraph(nx.DiGraph(incoming_graph_data=edges), root_node)
