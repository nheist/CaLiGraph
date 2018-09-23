import networkx as nx
from . import store as cat_store
from . import nlp as cat_nlp
import caligraph.util.nlp as nlp_util
import util
import numpy as np
from collections import Counter


class CategoryGraph:
    __DBP_TYPES_PROPERTY__ = 'dbp_types'

    def __init__(self, graph: nx.DiGraph, root_node: str):
        self.graph = graph
        self.root_node = root_node

    @property
    def statistics(self) -> str:
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        avg_indegree = np.mean([d for _, d in self.graph.in_degree])
        avg_outdegree = np.mean([d for _, d in self.graph.out_degree])
        dbp_typed_nodes_count = len({n for n in self.graph.nodes if self.dbp_types(n)})

        return '\n'.join([
            '{:^40}'.format('CATEGORY GRAPH STATISTICS'),
            '=' * 40,
            '{:>18} | {:>19}'.format('nodes', node_count),
            '{:>18} | {:>19}'.format('edges', edge_count),
            '{:>18} | {:>19.2f}'.format('in-degree', avg_indegree),
            '{:>18} | {:>19.2f}'.format('out-degree', avg_outdegree),
            '{:>18} | {:>19}'.format('dbp-typed nodes', dbp_typed_nodes_count),
        ])

    def predecessors(self, node: str) -> set:
        return set(self.graph.predecessors(node))

    def successors(self, node: str) -> set:
        return set(self.graph.successors(node))

    def depth(self, node: str) -> int:
        return nx.shortest_path_length(self.graph, source=self.root_node, target=node)

    def dbp_types(self, node: str) -> set:
        return self.graph.nodes(data=self.__DBP_TYPES_PROPERTY__)[node]

    def set_dbp_types(self, node: str, dbp_types: set):
        self.graph.nodes[node][self.__DBP_TYPES_PROPERTY__] = dbp_types

    def remove_unconnected(self):
        valid_nodes = set(nx.bfs_tree(self.graph, self.root_node))
        self._remove_all_nodes_except(valid_nodes)
        return self

    def append_unconnected(self):
        unconnected_root_nodes = {node for node in self.graph.nodes if not self.predecessors(node) and node != self.root_node}
        self.graph.add_edges_from([(self.root_node, node) for node in unconnected_root_nodes])
        return self

    def make_conceptual(self):
        categories = set(self.graph.nodes)
        # filtering maintenance categories
        categories = categories.difference(cat_store.get_maintenance_cats())
        # filtering administrative categories
        categories = {cat for cat in categories if not cat.endswith(('templates', 'navigational boxes'))}
        # filtering non-conceptual categories
        categories = {cat for cat in categories if cat_nlp.is_conceptual(cat)}
        # persisting spacy cache so that parsed categories are cached
        nlp_util.persist_cache()

        self._remove_all_nodes_except(categories | {self.root_node})
        return self

    def resolve_cycles(self):
        # remove all edges N1-->N2 of a cycle with depth(N1) > depth(N2)
        self._remove_cycle_edges_by_node_depth(lambda x, y: x > y)
        # remove all edges N1-->N2 of a cycle with depth(N1) >= depth(N2)
        self._remove_cycle_edges_by_node_depth(lambda x, y: x >= y)
        return self

    def _remove_cycle_edges_by_node_depth(self, comparator):
        edges_to_remove = set()
        for cycle in nx.simple_cycles(self.graph):
            node_depths = {node: self.depth(node) for node in cycle}
            for i in range(len(cycle)):
                current_edge = (cycle[i], cycle[(i+1) % len(cycle)])
                if comparator(node_depths[current_edge[0]], node_depths[current_edge[1]]):
                    util.get_logger().debug('Removing edge {}\nfrom cycle {}'.format(current_edge, cycle))
                    edges_to_remove.add(current_edge)
        self.graph.remove_edges_from(edges_to_remove)

    def compute_dbp_types(self):
        node_queue = [node for node in self.graph.nodes if not self.successors(node)]
        while node_queue:
            node = node_queue.pop(0)
            self._compute_dbp_types_for_node(node, node_queue)

        return self

    def _compute_dbp_types_for_node(self, node: str, node_queue: list) -> set:
        if self.dbp_types(node) is not None:
            return self.dbp_types(node)

        resource_type_distribution = cat_store.get_resource_type_distribution(node)
        resource_types = {t for t, probability in resource_type_distribution.items() if probability >= .5}

        children = self.successors(node)
        if children:
            child_type_count = sum([Counter(self._compute_dbp_types_for_node(c, node_queue)) for c in children], Counter())
            child_type_distribution = {t: count / len(children) for t, count in child_type_count.items()}
            child_types = {t for t, probability in child_type_distribution.items() if probability > .5}
            node_types = resource_types.intersection(child_types)
        else:
            node_types = resource_types

        if node_types:
            node_queue.extend(self.predecessors(node))

        self.set_dbp_types(node, node_types)
        return node_types

    def _remove_all_nodes_except(self, valid_nodes: set):
        invalid_nodes = set(self.graph.nodes).difference(valid_nodes)
        self.graph.remove_nodes_from(invalid_nodes)

    @classmethod
    def create_from_dbpedia(cls, root_node=None):
        edges = [(node, child) for node in cat_store.get_all_cats() for child in cat_store.get_children(node) if node != child]
        root_node = root_node if root_node else util.get_config('caligraph.category.root_node')
        return CategoryGraph(nx.DiGraph(incoming_graph_data=edges), root_node)
