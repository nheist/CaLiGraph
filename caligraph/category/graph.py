import networkx as nx
from . import store as cat_store
from . import nlp as cat_nlp
import caligraph.util.nlp as nlp_util
import util
import numpy as np

# note: use nx.simple_paths for depth measures


class CategoryGraph:
    def __init__(self, graph: nx.DiGraph, root_node: str):
        self.graph = graph
        self.root_node = root_node

    @property
    def statistics(self) -> str:
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        avg_indegree = np.mean([d for _, d in self.graph.in_degree])
        avg_outdegree = np.mean([d for _, d in self.graph.out_degree])

        return '\n'.join([
            '{:^40}'.format('CATEGORY GRAPH STATISTICS'),
            '=' * 40,
            '{:>18} | {:>19}'.format('nodes', node_count),
            '{:>18} | {:>19}'.format('edges', edge_count),
            '{:>18} | {:>19.2f}'.format('in-degree', avg_indegree),
            '{:>18} | {:>19.2f}'.format('out-degree', avg_outdegree)
        ])

    def predecessors(self, node: str) -> set:
        return set(self.graph.predecessors(node))

    def successors(self, node: str) -> set:
        return set(self.graph.successors(node))

    def depth(self, node: str) -> int:
        return nx.shortest_path_length(self.graph, source=self.root_node, target=node)

    def remove_unconnected(self):
        valid_nodes = set(nx.bfs_tree(self.graph, self.root_node))
        self._remove_all_nodes_except(valid_nodes)
        return self

    def append_unconnected(self):
        unconnected_root_nodes = {node for node in self.graph.nodes if len(self.predecessors(node)) == 0 and node != self.root_node}
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
        util.get_logger().debug('Removing edges N1-->N2 with depth(N1)>depth(N2)..')
        cycles_removed = True
        while cycles_removed:
            util.get_logger().debug('Starting new iteration..')
            cycles_removed = self._remove_cycle_edges_by_node_depth(lambda x, y: x > y)
        # remove all edges N1-->N2 of a cycle with depth(N1) >= depth(N2)
        util.get_logger().debug('Removing edges N1-->N2 with depth(N1)>=depth(N2)..')
        cycles_removed = True
        while cycles_removed:
            util.get_logger().debug('Starting new iteration..')
            cycles_removed = self._remove_cycle_edges_by_node_depth(lambda x, y: x >= y)
        return self

    def _remove_cycle_edges_by_node_depth(self, comparator) -> bool:
        edges_to_remove = set()
        for cycle in nx.simple_cycles(self.graph):
            node_depths = {node: self.depth(node) for node in cycle}
            for i in range(len(cycle)):
                current_edge = (cycle[i], cycle[(i+1) % len(cycle)])
                if comparator(node_depths[current_edge[0]], node_depths[current_edge[1]]):
                    edges_to_remove.add(current_edge)
                    util.get_logger().debug('Removing edge {}\nfrom cycle {}'.format(current_edge, cycle))

        if not edges_to_remove:
            return False
        self.graph.remove_edges_from(edges_to_remove)
        return True

    def _remove_all_nodes_except(self, valid_nodes: set):
        invalid_nodes = set(self.graph.nodes).difference(valid_nodes)
        self.graph.remove_nodes_from(invalid_nodes)

    @classmethod
    def create_from_dbpedia(cls, root_node=None):
        edges = [(node, child) for node in cat_store.get_all_cats() for child in cat_store.get_children(node) if node != child]
        root_node = root_node if root_node else util.get_config('caligraph.category.root_node')
        return CategoryGraph(nx.DiGraph(incoming_graph_data=edges), root_node)
