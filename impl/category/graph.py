import networkx as nx
from . import store as cat_store
from impl.util.base_graph import BaseGraph
from impl.category.conceptual import is_conceptual_category
import util
import numpy as np
from collections import defaultdict


class CategoryGraph(BaseGraph):
    PROPERTY_RESOURCE_TYPE_COUNTS = 'resource_type_counts'
    PROPERTY_DBP_TYPES = 'dbp_types'
    PROPERTY_MATERIALIZED_RESOURCES = 'materialized_resources'
    PROPERTY_MATERIALIZED_STATISTICS = 'materialized_statistics'

    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or util.get_config('category.root_category'))

    @property
    def categories(self) -> set:
        return self.nodes

    def add_edges(self, edges):
        # reset resource-dependent attributes that change due to the altered category tree structure
        parent_categories = {e[0] for e in edges}
        categories_to_reset = parent_categories | {a for cat in parent_categories for a in self.ancestors(cat)}
        for cat in categories_to_reset:
            self._reset_attr(cat, self.PROPERTY_MATERIALIZED_RESOURCES)
            self._reset_attr(cat, self.PROPERTY_MATERIALIZED_STATISTICS)

        super().add_edges(edges)

    @property
    def statistics(self) -> str:
        category_count = len(self.nodes)
        edge_count = len(self.edges)
        avg_indegree = np.mean([d for _, d in self.graph.in_degree])
        avg_outdegree = np.mean([d for _, d in self.graph.out_degree])

        return '\n'.join([
            '{:^40}'.format('CATEGORY GRAPH STATISTICS'),
            '=' * 40,
            '{:<30} | {:>7}'.format('categories', category_count),
            '{:<30} | {:>7}'.format('edges', edge_count),
            '{:<30} | {:>7.2f}'.format('in-degree', avg_indegree),
            '{:<30} | {:>7.2f}'.format('out-degree', avg_outdegree)
        ])

    def dbp_types(self, category: str) -> set:
        raise NotImplementedError('TODO: Implement using cat2ax.')

    def dbp_types_maybe(self, category: str) -> set:
        raise NotImplementedError('TODO: Implement using cat2ax.')

    def get_resources(self, category: str, materialized=False) -> set:
        if not self.has_node(category):
            raise Exception(f'Category {category} not in category graph.')

        resources = cat_store.get_resources(category)
        if not materialized:
            return resources

        if not self._get_attr(category, self.PROPERTY_MATERIALIZED_RESOURCES):
            materialized_resources = resources | {res for cat in self.successors(category) for res in self.get_resources(cat, materialized=True)}
            self._set_attr(category, self.PROPERTY_MATERIALIZED_RESOURCES, materialized_resources)
        return self._get_attr(category, self.PROPERTY_MATERIALIZED_RESOURCES)

    def get_statistics(self, category: str, materialized=False) -> dict:
        if not self.has_node(category):
            raise Exception(f'Category {category} not in category graph.')

        statistics = cat_store.get_statistics(category)
        if not materialized:
            return statistics

        if not self._get_attr(category, self.PROPERTY_MATERIALIZED_STATISTICS):
            resources = self.get_resources(category, materialized=True)
            type_counts = statistics['type_counts']
            property_counts = statistics['property_counts']
            for cat in self.successors(category):
                substats = self.get_statistics(cat, materialized=True)
                for t, c in substats['type_counts'].items():
                    type_counts[t] += c
                for p, c in substats['property_counts'].items():
                    property_counts[p] += c
            materialized_statistics = {
                'type_counts': type_counts,
                'type_frequencies': defaultdict(float, {t: t_count / len(resources) for t, t_count in type_counts.items()}),
                'property_counts': property_counts,
                'property_frequencies': defaultdict(float, {prop: p_count / len(resources) for prop, p_count in property_counts.items()}),
            }
            self._set_attr(category, self.PROPERTY_MATERIALIZED_STATISTICS, materialized_statistics)
        return self._get_attr(category, self.PROPERTY_MATERIALIZED_STATISTICS)

    @classmethod
    def create_from_dbpedia(cls):
        edges = [(cat, subcat) for cat in cat_store.get_categories() for subcat in cat_store.get_children(cat)]
        return CategoryGraph(nx.DiGraph(incoming_graph_data=edges))

    @classmethod
    def create_from_wikitaxonomy(cls):
        # TODO: include wikitaxonomy.py directly
        edges = list(util.load_cache('wikitaxonomy_edges'))
        return CategoryGraph(nx.DiGraph(incoming_graph_data=edges)).append_unconnected().remove_unconnected().resolve_cycles()

    # connectivity

    def remove_unconnected(self):
        valid_categories = {self.root_node} | nx.descendants(self.graph, self.root_node)
        self._remove_all_nodes_except(valid_categories)
        return self

    def append_unconnected(self):
        unconnected_root_categories = {cat for cat in self.nodes if not self.predecessors(cat) and cat != self.root_node}
        self.graph.add_edges_from([(self.root_node, cat) for cat in unconnected_root_categories])
        return self

    # conceptual categories

    def make_conceptual(self):
        categories = {c for c in self.nodes if cat_store.is_usable(c) and is_conceptual_category(c)}
        # clearing the graph of any invalid nodes
        self._remove_all_nodes_except(categories | {self.root_node})
        # appending all loose categories to the root node and removing the remaining self-referencing circular graphs
        self.append_unconnected().remove_unconnected()
        return self

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
            node_depths = {node: self._depth(node) for node in cycle}
            for i in range(len(cycle)):
                current_edge = (cycle[i], cycle[(i+1) % len(cycle)])
                if comparator(node_depths[current_edge[0]], node_depths[current_edge[1]]):
                    edges_to_remove.add(current_edge)
        self.graph.remove_edges_from(edges_to_remove)
