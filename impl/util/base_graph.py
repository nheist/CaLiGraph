import networkx as nx
import numpy as np
from collections import defaultdict


class BaseGraph:
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        self.graph = graph
        self.root_node = root_node

    def copy(self):
        return self.__class__(self.graph.copy(), root_node=self.root_node)

    @property
    def nodes(self) -> set:
        return set(self.graph)

    def has_node(self, node) -> bool:
        return node in self.graph

    def _add_nodes(self, nodes):
        self.graph.add_nodes_from(nodes)
        self._reset_node_indices()

    def _remove_nodes(self, nodes: set):
        self.graph.remove_nodes_from(nodes)
        self._reset_node_indices()

    def _remove_all_nodes_except(self, valid_nodes: set):
        self._remove_nodes(self.nodes.difference(valid_nodes))

    def _reset_node_indices(self):
        pass  # Triggered when anything changes in the node structure. Should be overriden by subclasses to clean up any outdated node indices.

    @property
    def edges(self) -> set:
        return set(self.graph.edges)

    def _add_edges(self, edges):
        self.graph.add_edges_from(edges)
        self._reset_edge_indices()

    def _remove_edges(self, edges):
        self.graph.remove_edges_from(edges)
        self._reset_edge_indices()

    def _remove_all_edges_except(self, valid_edges: set):
        invalid_edges = self.edges.difference(valid_edges)
        self._remove_edges(invalid_edges)

    def _reset_edge_indices(self):
        pass  # Triggered when anything changes in the edge structure. Should be overriden by subclasses to clean up any outdated edge indices.

    def parents(self, node: str) -> set:
        return set(self.graph.predecessors(node)) if self.graph.has_node(node) else set()

    def ancestors(self, node: str) -> set:
        return set(nx.ancestors(self.graph, node)) if self.graph.has_node(node) else set()

    def children(self, node: str) -> set:
        return set(self.graph.successors(node)) if self.graph.has_node(node) else set()

    def descendants(self, node: str) -> set:
        return set(nx.descendants(self.graph, node)) if self.graph.has_node(node) else set()

    def depth(self, node: str) -> int:
        try:
            return nx.shortest_path_length(self.graph, source=self.root_node, target=node)
        except nx.NetworkXNoPath:
            return -1

    def depths(self) -> dict:
        return defaultdict(lambda: -1, nx.shortest_path_length(self.graph, source=self.root_node))

    def _get_attr(self, node, attr):
        return self.graph.nodes(data=attr)[node]

    def _set_attr(self, node, attr, val):
        self.graph.nodes[node][attr] = val

    def _reset_attr(self, node, attr):
        if attr in self.graph.nodes[node]:
            del self.graph.nodes[node][attr]

    @property
    def statistics(self) -> str:
        type_count = len(self.nodes)
        edge_count = len(self.edges)
        avg_degree = np.mean([d for _, d in self.graph.in_degree])

        return '\n'.join([
            '{:^40}'.format('STATISTICS'),
            '=' * 40,
            '{:<30} | {:>7}'.format('nodes', type_count),
            '{:<30} | {:>7}'.format('edges', edge_count),
            '{:<30} | {:>7.2f}'.format('degree', avg_degree),
            ])
