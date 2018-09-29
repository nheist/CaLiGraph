import networkx as nx
from . import store as cat_store
from . import nlp as cat_nlp
import caligraph.dbpedia.store as dbp_store
import util
import numpy as np
from collections import Counter
import math


class CategoryGraph:
    PROPERTY_DBP_TYPES = 'dbp_types'
    PROPERTY_RESOURCE_TYPE_COUNTS = 'resource_type_counts'

    def __init__(self, graph: nx.DiGraph, root_category: str):
        self.graph = graph
        self.root_category = root_category

    @property
    def statistics(self) -> str:
        category_count = len(self.categories)
        edge_count = self.graph.number_of_edges()
        avg_indegree = np.mean([d for _, d in self.graph.in_degree])
        avg_outdegree = np.mean([d for _, d in self.graph.out_degree])

        dbp_typed_categories = {cat for cat in self.categories if self.dbp_types(cat)}
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
    
    @property
    def categories(self) -> set:
        return set(self.graph.nodes)

    def dbp_types(self, category: str) -> set:
        return self._get_attr(category, self.PROPERTY_DBP_TYPES)

    def copy(self):
        return CategoryGraph(self.graph.copy(), self.root_category)

    @classmethod
    def create_from_dbpedia(cls, root_category=None):
        edges = [(cat, subcat) for cat in cat_store.get_all_cats() for subcat in cat_store.get_children(cat) if cat != subcat]
        root_category = root_category or util.get_config('caligraph.category.root_category')
        return CategoryGraph(nx.DiGraph(incoming_graph_data=edges), root_category)

    def _predecessors(self, category: str) -> set:
        return set(self.graph.predecessors(category))

    def _successors(self, category: str) -> set:
        return set(self.graph.successors(category))

    def _depth(self, category: str) -> int:
        return nx.shortest_path_length(self.graph, source=self.root_category, target=category)

    def _get_attr(self, category, attr):
        return self.graph.nodes(data=attr)[category]

    def _set_attr(self, category, attr, val):
        self.graph.nodes[category][attr] = val

    # connectivity

    def remove_unconnected(self):
        valid_categories = {self.root_category} | nx.descendants(self.graph, self.root_category)
        self._remove_all_categories_except(valid_categories)
        return self

    def append_unconnected(self):
        unconnected_root_categories = {cat for cat in self.categories if not self._predecessors(cat) and cat != self.root_category}
        self.graph.add_edges_from([(self.root_category, cat) for cat in unconnected_root_categories])
        return self

    # conceptual categories

    def make_conceptual(self):
        categories = self.categories
        # filtering maintenance categories
        categories = categories.difference(cat_store.get_maintenance_cats())
        # filtering administrative categories
        categories = {cat for cat in categories if not cat.endswith(('templates', 'navigational boxes'))}
        # filtering non-conceptual categories
        categories = {cat for cat in categories if cat_nlp.is_conceptual(cat)}
        # persisting spacy cache so that parsed categories are cached
        cat_nlp.persist_cache()
        # clearing the graph of any invalid nodes
        self._remove_all_categories_except(categories | {self.root_category})
        # appending all loose categories to the root node and removing the remaining self-referencing circular graphs
        self.append_unconnected().remove_unconnected()
        return self

    def _remove_all_categories_except(self, valid_categories: set):
        invalid_categories = self.categories.difference(valid_categories)
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
    RESOURCE_TYPE_RATIO_THRESHOLD = .5
    RESOURCE_TYPE_COUNT_THRESHOLD = 1  # >1 leads to high loss in recall and only moderate increase of precision -> measure is too restrictive
    EXCLUDE_UNTYPED_RESOURCES = True  # False leads to a moderate loss of precision and a high loss of recall
    FILTER_LOW_EVIDENCE_TYPES = False
    CHILDREN_TYPE_RATIO_THRESHOLD = .5

    def assign_dbp_types(self):
        self._assign_resource_type_counts()
        category_queue = [cat for cat in self.categories if not self._successors(cat)]
        while category_queue:
            cat = category_queue.pop(0)
            self._compute_dbp_types_for_category(cat, category_queue)

        return self

    def _assign_resource_type_counts(self):
        categories = {cat for cat in self.categories if not self._get_attr(cat, self.PROPERTY_RESOURCE_TYPE_COUNTS)}
        for cat in categories:
            resources_types = {r: dbp_store.get_transitive_types(r) for r in cat_store.get_resources(cat)}
            resource_count = len(resources_types)
            typed_resource_count = len({r for r, types in resources_types.items() if types})
            types_count = sum([Counter(types) for _, types in resources_types.items()], Counter())
            self._set_attr(cat, self.PROPERTY_RESOURCE_TYPE_COUNTS, {'count': resource_count, 'typed_count': typed_resource_count, 'types': types_count})

    def _compute_dbp_types_for_category(self, cat: str, category_queue: list) -> set:
        if self.dbp_types(cat) is not None:
            return self.dbp_types(cat)

        resource_types = self._compute_resource_types_for_category(cat)

        # compare with child types
        children_with_types = {cat: self._compute_dbp_types_for_category(cat, category_queue) for cat in self._successors(cat)}
        if len(children_with_types) == 0:
            category_types = resource_types
        else:
            child_type_counts = sum([Counter(types) for types in children_with_types.values()], Counter())
            if resource_types:
                child_types = set(child_type_counts.keys())
                category_types = resource_types.intersection(child_types)
            else:
                child_type_distribution = {t: count / len(children_with_types) for t, count in child_type_counts.items()}
                category_types = {t for t, probability in child_type_distribution.items() if probability > self.CHILDREN_TYPE_RATIO_THRESHOLD}

        # remove any disjoint type assignments
        category_types = category_types.difference({dt for t in category_types for dt in dbp_store.get_disjoint_types(t)})

        if category_types:
            category_queue.extend(self._predecessors(cat))

        self._set_attr(cat, self.PROPERTY_DBP_TYPES, category_types)
        return category_types

    def _compute_resource_types_for_category(self, cat: str) -> set:
        resource_type_counts = self._get_attr(cat, self.PROPERTY_RESOURCE_TYPE_COUNTS)
        # filter types due to absolute counts
        resource_type_counts['types'] = {t: t_count for t, t_count in resource_type_counts['types'].items() if t_count >= self.RESOURCE_TYPE_COUNT_THRESHOLD}
        # filter types due to counts relative to ontology depth
        if self.FILTER_LOW_EVIDENCE_TYPES:
            resource_type_counts['types'] = {t: t_count for t, t_count in resource_type_counts['types'].items() if t_count >= math.log2(dbp_store.get_type_depth(t))}

        resource_count = resource_type_counts['typed_count' if self.EXCLUDE_UNTYPED_RESOURCES else 'count']
        resource_type_distribution = {t: t_count / resource_count for t, t_count in resource_type_counts['types'].items()}
        return {t for t, probability in resource_type_distribution.items() if probability > self.RESOURCE_TYPE_RATIO_THRESHOLD}
