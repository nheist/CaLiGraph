import networkx as nx
from . import store as cat_store
from . import nlp as cat_nlp
import caligraph.dbpedia.store as dbp_store
import caligraph.dbpedia.util as dbp_util
import util
import numpy as np
from collections import Counter


class CategoryGraph:
    __DBP_TYPES_PROPERTY__ = 'dbp_types'
    __RESOURCE_TYPE_DISTRIBUTION_PROPERTY__ = 'resource_type_distribution'

    def __init__(self, graph: nx.DiGraph, root_node: str):
        self.graph = graph
        self.root_node = root_node

    @property
    def statistics(self) -> str:
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        avg_indegree = np.mean([d for _, d in self.graph.in_degree])
        avg_outdegree = np.mean([d for _, d in self.graph.out_degree])

        dbp_typed_nodes = {n for n in self.graph.nodes if self.dbp_types(n)}
        dbp_typed_node_count = len(dbp_typed_nodes)
        avg_dbp_types = np.mean([len(self.dbp_types(n)) for n in dbp_typed_nodes]) if dbp_typed_nodes else 0

        return '\n'.join([
            '{:^40}'.format('CATEGORY GRAPH STATISTICS'),
            '=' * 40,
            '{:<30} | {:>7}'.format('nodes', node_count),
            '{:<30} | {:>7}'.format('edges', edge_count),
            '{:<30} | {:>7.2f}'.format('in-degree', avg_indegree),
            '{:<30} | {:>7.2f}'.format('out-degree', avg_outdegree),
            '{:<30} | {:>7}'.format('dbp-typed nodes', dbp_typed_node_count),
            '{:<30} | {:>7.2f}'.format('dbp-types per node', avg_dbp_types)
        ])

    def predecessors(self, node: str) -> set:
        return set(self.graph.predecessors(node))

    def successors(self, node: str) -> set:
        return set(self.graph.successors(node))

    def depth(self, node: str) -> int:
        return nx.shortest_path_length(self.graph, source=self.root_node, target=node)

    def dbp_types(self, node: str) -> set:
        return self._get_attr(node, self.__DBP_TYPES_PROPERTY__)

    def _get_attr(self, node, attr):
        return self.graph.nodes(data=attr)[node]

    def _set_attr(self, node, attr, val):
        self.graph.nodes[node][attr] = val

    def copy(self):
        return CategoryGraph(self.graph.copy(), self.root_node)

    @classmethod
    def create_from_dbpedia(cls, root_node=None):
        edges = [(node, child) for node in cat_store.get_all_cats() for child in cat_store.get_children(node) if node != child]
        root_node = root_node if root_node else util.get_config('caligraph.category.root_node')
        return CategoryGraph(nx.DiGraph(incoming_graph_data=edges), root_node)

    # connectivity

    def remove_unconnected(self):
        valid_nodes = set(nx.bfs_tree(self.graph, self.root_node))
        self._remove_all_nodes_except(valid_nodes)
        return self

    def append_unconnected(self):
        unconnected_root_nodes = {node for node in self.graph.nodes if not self.predecessors(node) and node != self.root_node}
        self.graph.add_edges_from([(self.root_node, node) for node in unconnected_root_nodes])
        return self

    # conceptual categories

    def make_conceptual(self):
        categories = set(self.graph.nodes)
        # filtering maintenance categories
        categories = categories.difference(cat_store.get_maintenance_cats())
        # filtering administrative categories
        categories = {cat for cat in categories if not cat.endswith(('templates', 'navigational boxes'))}
        # filtering non-conceptual categories
        categories = {cat for cat in categories if cat_nlp.is_conceptual(cat)}
        # persisting spacy cache so that parsed categories are cached
        cat_nlp.persist_cache()

        self._remove_all_nodes_except(categories | {self.root_node})
        return self

    def _remove_all_nodes_except(self, valid_nodes: set):
        invalid_nodes = set(self.graph.nodes).difference(valid_nodes)
        self.graph.remove_nodes_from(invalid_nodes)

    # cycles

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
                    edges_to_remove.add(current_edge)
        self.graph.remove_edges_from(edges_to_remove)

    # dbp-types
    RESOURCE_TYPE_THRESHOLD = .5
    EXCLUDE_UNTYPED_RESOURCES = False
    CHILDREN_TYPE_THRESHOLD = .5
    EXCLUDE_UNTYPED_CHILDREN = False

    def compute_dbp_types(self):
        node_queue = [node for node in self.graph.nodes if not self.successors(node)]
        while node_queue:
            node = node_queue.pop(0)
            self._compute_dbp_types_for_node(node, node_queue)

        return self

    def _compute_dbp_types_for_node(self, node: str, node_queue: list) -> set:
        if self.dbp_types(node) is not None:
            return self.dbp_types(node)

        resource_type_distribution = self._compute_resource_type_distribution(node)
        resource_types = {t for t, probability in resource_type_distribution.items() if probability >= self.RESOURCE_TYPE_THRESHOLD}

        children = self.successors(node)
        children_types = {c: self._compute_dbp_types_for_node(c, node_queue) for c in children}
        child_count = len({c for c, types in children_types.items() if types} if self.EXCLUDE_UNTYPED_CHILDREN else children)
        if children:
            child_type_count = sum([Counter(types) for types in children_types.values()], Counter())
            child_type_distribution = {t: count / child_count for t, count in child_type_count.items()}
            child_types = {t for t, probability in child_type_distribution.items() if probability > self.CHILDREN_TYPE_THRESHOLD}
            node_types = resource_types.intersection(child_types) if resource_types else child_types
        else:
            node_types = resource_types

        if node_types:
            node_queue.extend(self.predecessors(node))

        self._set_attr(node, self.__DBP_TYPES_PROPERTY__, node_types)
        return node_types

    def assign_dbp_types(self):
        for node in self.graph.nodes:
            resource_type_distribution = self._compute_resource_type_distribution(node)
            self._set_attr(node, self.__RESOURCE_TYPE_DISTRIBUTION_PROPERTY__, resource_type_distribution)

            dbp_types = {t for t, probability in resource_type_distribution.items() if probability >= self.RESOURCE_TYPE_THRESHOLD}
            self._set_attr(node, self.__DBP_TYPES_PROPERTY__, dbp_types)

        return self

    def _compute_resource_type_distribution(self, node: str) -> dict:
        resources_types = {r: dbp_store.get_transitive_types(r) for r in cat_store.get_resources(node)}
        resource_count = len({r for r, types in resources_types.items() if types} if self.EXCLUDE_UNTYPED_RESOURCES else resources_types)
        resource_type_count = sum([Counter(types) for r, types in resources_types.items()], Counter())
        return {t: count / resource_count for t, count in resource_type_count.items()}
