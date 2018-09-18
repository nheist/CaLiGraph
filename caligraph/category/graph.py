import networkx as nx
from . import store as cat_store


class CategoryGraph:
    def __init__(self, graph: nx.DiGraph, root_node: str):
        self.graph = graph
        self.root_node = root_node

    def get_node_count(self) -> int:
        return self.graph.number_of_nodes()

    def get_edge_count(self) -> int:
        return self.graph.number_of_edges()

    def get_connected_graph(self):
        return CategoryGraph(nx.bfs_tree(self.graph, self.root_node), self.root_node)

    @classmethod
    def create_from_dbpedia(cls):
        edges = [(node, child) for node in cat_store.get_all_cats() for child in cat_store.get_children(node)]
        return nx.DiGraph(incoming_graph_data=edges)
