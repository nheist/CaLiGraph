import networkx as nx
from . import store as cat_store
from . import nlp as cat_nlp
import impl.dbpedia.store as dbp_store
from impl.util.base_graph import BaseGraph
import impl.util.rdf as rdf_util
import util
import numpy as np
import math
import pandas as pd
import random


class CategoryGraph(BaseGraph):
    PROPERTY_RESOURCE_TYPE_COUNTS = 'resource_type_counts'
    PROPERTY_DBP_TYPES = 'dbp_types'
    PROPERTY_MATERIALIZED_RESOURCES = 'materialized_resources'

    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or util.get_config('category.root_category'))

    @property
    def statistics(self) -> str:
        category_count = len(self.nodes)
        edge_count = len(self.edges)
        avg_indegree = np.mean([d for _, d in self.graph.in_degree])
        avg_outdegree = np.mean([d for _, d in self.graph.out_degree])

        dbp_typed_categories = {cat for cat in self.nodes if self.dbp_types(cat)}
        dbp_typed_category_count = len(dbp_typed_categories)
        avg_dbp_types = np.mean([len(self.dbp_types(c)) for c in dbp_typed_categories]) if dbp_typed_categories else 0

        return '\n'.join([
            '{:^40}'.format('CATEGORY GRAPH STATISTICS'),
            '=' * 40,
            '{:<30} | {:>7}'.format('categories', category_count),
            '{:<30} | {:>7}'.format('edges', edge_count),
            '{:<30} | {:>7.2f}'.format('in-degree', avg_indegree),
            '{:<30} | {:>7.2f}'.format('out-degree', avg_outdegree),
            '{:<30} | {:>7}'.format('dbp-typed categories', dbp_typed_category_count),
            '{:<30} | {:>7.2f}'.format('dbp-types per category', avg_dbp_types)
        ])

    def dbp_types(self, category: str) -> set:
        return self._get_attr(category, self.PROPERTY_DBP_TYPES) if category in self.nodes else set()

    def get_materialized_resources(self, category: str) -> set:
        if category not in self.nodes:
            return cat_store.get_resources(category)

        if not self._get_attr(category, self.PROPERTY_MATERIALIZED_RESOURCES):
            materialized_resources = cat_store.get_resources(category) | {res for cat in self.successors(category) for res in self.get_materialized_resources(cat)}
            self._set_attr(category, self.PROPERTY_MATERIALIZED_RESOURCES, materialized_resources)
        return self._get_attr(category, self.PROPERTY_MATERIALIZED_RESOURCES)

    @classmethod
    def create_from_dbpedia(cls):
        edges = [(cat, subcat) for cat in cat_store.get_all_cats() for subcat in cat_store.get_children(cat) if cat != subcat]
        return CategoryGraph(nx.DiGraph(incoming_graph_data=edges))

    # connectivity

    def remove_unconnected(self):
        valid_categories = {self.root_node} | nx.descendants(self.graph, self.root_node)
        self._remove_all_categories_except(valid_categories)
        return self

    def append_unconnected(self):
        unconnected_root_categories = {cat for cat in self.nodes if not self.predecessors(cat) and cat != self.root_node}
        self.graph.add_edges_from([(self.root_node, cat) for cat in unconnected_root_categories])
        return self

    # conceptual categories

    def make_conceptual(self):
        categories = self.nodes
        # filtering maintenance categories
        categories = categories.difference(cat_store.get_maintenance_cats())
        # filtering administrative categories
        categories = {cat for cat in categories if not cat.endswith(('templates', 'navigational boxes'))}
        # filtering non-conceptual categories
        categories = {cat for cat in categories if cat_nlp.is_conceptual(cat)}
        # persisting spacy cache so that parsed categories are cached
        cat_nlp.persist_cache()
        # clearing the graph of any invalid nodes
        self._remove_all_categories_except(categories | {self.root_node})
        # appending all loose categories to the root node and removing the remaining self-referencing circular graphs
        self.append_unconnected().remove_unconnected()
        return self

    def _remove_all_categories_except(self, valid_categories: set):
        invalid_categories = self.nodes.difference(valid_categories)
        self.graph.remove_nodes_from(invalid_categories)

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

    # dbp-types

    def assign_dbp_types(self):
        self._assign_resource_type_counts()
        category_queue = [cat for cat in self.nodes if not self.successors(cat)]
        while category_queue:
            cat = category_queue.pop(0)
            self._compute_dbp_types_for_category(cat, category_queue)

        return self

    def _assign_resource_type_counts(self):
        categories = {cat for cat in self.nodes if not self._get_attr(cat, self.PROPERTY_RESOURCE_TYPE_COUNTS)}
        for cat in categories:
            resources_types = {r: dbp_store.get_transitive_types(r) for r in cat_store.get_resources(cat)}
            resource_count = len(resources_types)
            typed_resource_count = len({r for r, types in resources_types.items() if types})
            types_count = rdf_util.create_count_dict(resources_types.values())
            self._set_attr(cat, self.PROPERTY_RESOURCE_TYPE_COUNTS, {'count': resource_count, 'typed_count': typed_resource_count, 'types': types_count})

    def _compute_dbp_types_for_category(self, cat: str, category_queue: list, processed_cats=0) -> set:
        if self.dbp_types(cat) is not None:
            return self.dbp_types(cat)

        resource_types = self._compute_resource_types_for_category(cat)

        # compare with child types
        children_with_types = {cat: self._compute_dbp_types_for_category(cat, category_queue, processed_cats) for cat in self.successors(cat)}
        if len(children_with_types) == 0:
            category_types = resource_types
        else:
            child_type_counts = rdf_util.create_count_dict(children_with_types.values())
            if util.get_config('category.dbp_types.prefer_resource_types') and resource_types:
                child_types = set(child_type_counts.keys())
                category_types = resource_types.intersection(child_types)
            else:
                child_type_distribution = {t: count / len(children_with_types) for t, count in child_type_counts.items()}
                child_type_ratio = util.get_config('category.dbp_types.child_type_ratio')
                category_types = {t for t, probability in child_type_distribution.items() if probability > child_type_ratio}

        # remove any disjoint type assignments
        category_types = category_types.difference({dt for t in category_types for dt in dbp_store.get_disjoint_types(t)})

        if util.get_config('category.dbp_types.apply_impure_type_filtering'):
            category_types = self._filter_impure_types(cat, category_types)

        if category_types:
            category_queue.extend(self.predecessors(cat))

        self._set_attr(cat, self.PROPERTY_DBP_TYPES, category_types)

        processed_cats += 1
        if processed_cats % 100 == 0:
            util.get_logger().debug(f'DBP-TYPE-ASSIGNMENT: Processed {processed_cats} categories.')
        return category_types

    def _compute_resource_types_for_category(self, cat: str) -> set:
        resource_type_counts = self._get_attr(cat, self.PROPERTY_RESOURCE_TYPE_COUNTS)
        # filter type counts relative to ontology depth
        if util.get_config('category.dbp_types.apply_type_depth_smoothing'):
            resource_type_counts['types'] = {t: t_count for t, t_count in resource_type_counts['types'].items() if t_count >= math.log2(dbp_store.get_type_depth(t))}

        resource_count = resource_type_counts['typed_count' if util.get_config('category.dbp_types.exclude_untyped_resources') else 'count']
        resource_type_distribution = {t: t_count / resource_count for t, t_count in resource_type_counts['types'].items()}
        resource_type_ratio = util.get_config('category.dbp_types.resource_type_ratio')
        return {t for t, probability in resource_type_distribution.items() if probability > resource_type_ratio}

    def _filter_impure_types(self, category: str, category_types: set) -> set:
        resources_with_types = {r: dbp_store.get_transitive_types(r) for r in cat_store.get_resources(category)}
        if not resources_with_types:
            return category_types

        pure_category_types = set()
        type_purity_threshold = util.get_config('category.dbp_types.type_purity_threshold')
        for cat_type in category_types:
            impure_resource_count = len({r for r, types in resources_with_types.items() if any(dbp_store.get_cooccurrence_frequency(cat_type, t) == 0 for t in types)})
            if (1 - impure_resource_count / len(resources_with_types)) >= type_purity_threshold:
                pure_category_types.add(cat_type)

        return pure_category_types

    # dbp-types (evaluation)

    def create_dbp_type_sample(self):
        self._assign_resource_type_counts()
        category_sample = random.sample(self.nodes, 1000)
        category_types = [{'cat': cat, 'dbp_type': dbp_type} for cat in category_sample for dbp_type in self._get_attr(cat, self.PROPERTY_RESOURCE_TYPE_COUNTS)['types']]
        pd.DataFrame(data=category_types).to_csv(util.get_results_file('results.catgraph.dbp_type_sample'), index=False)

    # taxonomy

    def make_taxonomy(self):
        # remove nodes / edges with missing type / type mismatch
        self.graph.remove_nodes_from({node for node in self.nodes if not self.dbp_types(node)})
        self.graph.remove_edges_from({e for e in self.graph.edges if not self.dbp_types(e[0]).intersection(self.dbp_types(e[1]))})
        # reinsert root node and connect with top-level nodes
        self.graph.add_node(self.root_node)
        self.graph.add_edges_from({(self.root_node, node) for node in self.graph.nodes if not self.graph.predecessors(node)})

        return self
