import networkx as nx
from collections import defaultdict
import copy
from typing import Iterator


class BaseGraph:
    """A simple graph with nodes and edges."""

    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        self.graph = graph
        self.root_node = root_node

    def copy(self):
        new_self = self.__class__(copy.deepcopy(self.graph), root_node=self.root_node)
        return new_self

    @property
    def nodes(self) -> set:
        return set(self.graph)

    @property
    def content_nodes(self) -> set:
        return set(self.graph).difference({self.root_node})

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
        """This method is triggered when anything changes in the node structure.
        Should be overridden by subclasses to clean up any outdated node indices.
        """
        pass

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
        """This method is triggered when anything changes in the node structure.
        Should be overridden by subclasses to clean up any outdated edge indices.
        """
        pass

    def parents(self, node: str) -> set:
        return set(self.graph.predecessors(node)) if self.graph.has_node(node) else set()

    def ancestors(self, node: str) -> set:
        return set(nx.ancestors(self.graph, node)) if self.graph.has_node(node) else set()

    def children(self, node: str) -> set:
        return set(self.graph.successors(node)) if self.graph.has_node(node) else set()

    def descendants(self, node: str) -> set:
        return set(nx.descendants(self.graph, node)) if self.graph.has_node(node) else set()

    def depth(self, node: str) -> int:
        """Returns the length of the shortest path from a given node to the root node (or -1 if none exists)."""
        try:
            return nx.shortest_path_length(self.graph, source=self.root_node, target=node)
        except nx.NetworkXNoPath:
            return -1

    def depths(self) -> dict:
        """Returns the lengths of the shortest path from all nodes of the graph to the root node."""
        return defaultdict(lambda: -1, nx.shortest_path_length(self.graph, source=self.root_node))

    def _get_attr(self, node, attr):
        return self.graph.nodes(data=attr)[node]

    def _set_attr(self, node, attr, val):
        self.graph.nodes[node][attr] = val

    def _reset_attr(self, node, attr):
        if attr in self.graph.nodes[node]:
            del self.graph.nodes[node][attr]

    def traverse_nodes_topdown(self) -> Iterator:
        """Traverse nodes in such a way that all parents of a node have been traversed before the node itself.
        The graph has to be fully-connected and cycle-free.
        """
        visited_nodes = set()
        node_queue = [self.root_node]
        while node_queue:
            node = node_queue.pop(0)
            if node in visited_nodes:
                continue
            if not self.parents(node).issubset(visited_nodes):
                node_queue.append(node)
                continue
            visited_nodes.add(node)
            node_queue.extend(self.children(node))
            yield node

    def traverse_edges_topdown(self) -> Iterator:
        """Traverse edges in such a way that the target of an edge must have been
        the source of an edge at least once before (except for the root node).
        The graph has to be fully-connected and cycle-free.
        """
        for parent in self.traverse_nodes_topdown():
            for child in self.children(parent):
                yield parent, child

    def traverse_nodes_bottomup(self) -> Iterator:
        """Traverse nodes in such a way that all children of a node have been traversed before the node itself.
        The graph has to be fully-connected and cycle-free.
        """
        visited_nodes = set()
        node_queue = [n for n in self.nodes if not self.children(n)]
        while node_queue:
            node = node_queue.pop(0)
            if node in visited_nodes:
                continue
            if not self.children(node).issubset(visited_nodes):
                node_queue.append(node)
                continue
            visited_nodes.add(node)
            node_queue.extend(self.parents(node))
            yield node
