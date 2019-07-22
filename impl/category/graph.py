import networkx as nx
from typing import Optional
import impl.category.store as cat_store
import impl.category.wikitaxonomy as cat_wikitax
from impl.category.conceptual import is_conceptual_category
from impl.util.base_graph import BaseGraph
import util


class CategoryGraph(BaseGraph):
    # initialisations
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or util.get_config('category.root_category'))

        self._node_by_category = None

    # node categories
    ATTRIBUTE_CATEGORY = 'attribute_category'

    def get_node_for_category(self, category: str) -> Optional[str]:
        if self._node_by_category is None:  # initialise category-node index if not existing
            self._node_by_category = {cat: node for node in self.nodes for cat in self.get_categories(node)}
        return self._node_by_category[category] if category in self._node_by_category else None

    def get_categories(self, node: str) -> set:
        if not self.has_node(node):
            raise Exception(f'Node {node} not in category graph.')
        return self._get_attr(node, self.ATTRIBUTE_CATEGORY)

    def _set_categories(self, node: str, categories: set):
        if not self.has_node(node):
            raise Exception(f'Node {node} not in category graph.')
        self._set_attr(node, self.ATTRIBUTE_CATEGORY, categories)
        self._node_by_category = None  # reset category-node index due to changes

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
    # TODO: refactor most graph creation methods out of this class

    @classmethod
    def create_from_dbpedia(cls):
        edges = [(cat, subcat) for cat in cat_store.get_categories() for subcat in cat_store.get_children(cat)]
        graph = CategoryGraph(nx.DiGraph(incoming_graph_data=edges))
        for node in graph.nodes:
            graph._set_categories(node, {node})
        return graph

    # ensure connectivity

    def remove_unconnected(self):
        valid_categories = {self.root_node} | nx.descendants(self.graph, self.root_node)
        self._remove_all_nodes_except(valid_categories)
        return self

    def append_unconnected(self):
        unconnected_root_categories = {cat for cat in self.nodes if not self.parents(cat) and cat != self.root_node}
        self._add_edges([(self.root_node, cat) for cat in unconnected_root_categories])
        return self

    # filter for conceptual categories

    def make_conceptual(self):
        categories = {c for c in self.nodes if cat_store.is_usable(c) and is_conceptual_category(c)}
        # clearing the graph of any invalid nodes
        self._remove_all_nodes_except(categories | {self.root_node})
        # appending all loose categories to the root node and removing the remaining self-referencing circular graphs
        self.append_unconnected().remove_unconnected()
        return self

    # apply wikitaxonomy

    def apply_wikitaxonomy(self):
        valid_edges = cat_wikitax.get_valid_edges(self.edges)
        valid_nodes = {e[0] for e in valid_edges} | {e[1] for e in valid_edges} | {self.root_node}
        # remove all non-wikitaxonomy nodes
        self._remove_all_nodes_except(valid_nodes)
        # remove all non-wikitaxonomy edges
        self._remove_all_edges_except(valid_edges)
        # appending all loose categories to the root node and removing the remaining self-referencing circular graphs
        self.append_unconnected().remove_unconnected()
        return self

    # resolve cycles

    def resolve_cycles(self):
        # remove all edges N1-->N2 of a cycle with depth(N1) > depth(N2)
        self._remove_cycle_edges_by_node_depth(lambda x, y: x > y)
        # remove all edges N1-->N2 of a cycle with depth(N1) >= depth(N2)
        self._remove_cycle_edges_by_node_depth(lambda x, y: x >= y)
        return self

    def _remove_cycle_edges_by_node_depth(self, comparator):
        edges_to_remove = set()
        for cycle in nx.simple_cycles(self.graph):
            node_depths = {node: self._depth(node) for node in cycle}
            for i in range(len(cycle)):
                current_edge = (cycle[i], cycle[(i+1) % len(cycle)])
                if comparator(node_depths[current_edge[0]], node_depths[current_edge[1]]):
                    edges_to_remove.add(current_edge)
        self._remove_edges(edges_to_remove)

    # merge category nodes

    def merge_nodes(self):
        something_changed = True
        while something_changed:
            something_changed = False
            traversed_nodes = set()
            for node, _ in nx.traversal.bfs_edges(self.graph, self.root_node):
                if node in traversed_nodes:
                    continue
                traversed_nodes.add(node)

                children_to_merge = {c for c in self.children(node) if self._should_merge_nodes(node, c)}
                if children_to_merge:
                    util.get_logger().debug(f'Merging nodes "{children_to_merge}" into {node}.')
                    node_new_children = {child for cat in children_to_merge for child in self.children(cat)}
                    edges_to_add = {(node, child) for child in node_new_children}
                    edges_to_remove = {(node, old_child) for old_child in children_to_merge}
                    self._remove_edges(edges_to_remove)
                    self._add_edges(edges_to_add)

                    node_categories = self.get_categories(node) | children_to_merge
                    self._set_categories(node, node_categories)

                    something_changed = True
        return self

    @staticmethod
    def _should_merge_nodes(parent: str, child: str) -> bool:
        return child.startswith(parent) and child[len(parent):].startswith('_by_')
