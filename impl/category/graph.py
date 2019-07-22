import networkx as nx
from typing import Optional
import impl.category.util as cat_util
import impl.category.store as cat_store
from impl.category.conceptual import is_conceptual_category
from impl.util.hierarchy_graph import HierarchyGraph
import util


class CategoryGraph(HierarchyGraph):
    # initialisations
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or util.get_config('category.root_category'))

    # node categories
    def get_node_for_category(self, category: str) -> Optional[str]:
        return self.get_node_for_part(category)

    def get_categories(self, node: str) -> set:
        return self.get_parts(node)

    def _set_categories(self, node: str, categories: set):
        self._set_parts(node, categories)

    # node resources

    def get_resources(self, node: str) -> set:
        if not self.has_node(node):
            raise Exception(f'Node {node} not in category graph.')
        return {res for cat in self.get_categories(node) for res in cat_store.get_resources(cat)}

    def get_statistics(self, node: str) -> dict:
        if not self.has_node(node):
            raise Exception(f'Node {node} not in category graph.')
        return cat_store.get_statistics(node)

    # GRAPH CREATION

    @classmethod
    def create_from_dbpedia(cls):
        edges = [(cat, subcat) for cat in cat_store.get_categories() for subcat in cat_store.get_children(cat)]
        graph = CategoryGraph(nx.DiGraph(incoming_graph_data=edges))

        for node in graph.nodes:
            graph._set_name(node, cat_util.category2name(node))
            graph._set_parts(node, {node})

        return graph

    # filter for conceptual categories

    def make_conceptual(self):
        categories = {c for c in self.nodes if cat_store.is_usable(c) and is_conceptual_category(c)}
        # clearing the graph of any invalid nodes
        self._remove_all_nodes_except(categories | {self.root_node})
        # appending all loose categories to the root node and removing the remaining self-referencing circular graphs
        self.append_unconnected().remove_unconnected()
        return self
