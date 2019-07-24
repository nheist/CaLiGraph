from typing import Optional
import networkx as nx
import util
import impl.util.nlp as nlp_util
import impl.util.hypernymy as hypernymy_util
from impl.util.base_graph import BaseGraph
import random
from collections import defaultdict


class HierarchyGraph(BaseGraph):
    # initialisations
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node)
        self._node_by_name = None
        self._node_by_part = None

    def _check_node_exists(self, node: str):
        if not self.has_node(node):
            raise Exception(f'Node {node} not in graph.')

    # node attribute definitions
    ATTRIBUTE_NAME = 'attribute_name'
    ATTRIBUTE_PARTS = 'attribute_parts'

    def get_name(self, node: str) -> str:
        self._check_node_exists(node)
        return self._get_attr(node, self.ATTRIBUTE_NAME)

    def _set_name(self, node: str, name: str):
        self._check_node_exists(node)
        self._set_attr(node, self.ATTRIBUTE_NAME, name)
        self._node_by_name = None  # reset name-to-node index due to changes

    def get_node_by_name(self, name: str) -> Optional[str]:
        if self._node_by_name is None:  # initialise name-to-node index if not existing
            self._node_by_name = {self.get_name(node): node for node in self.nodes}
        return self._node_by_name[name] if name in self._node_by_name else None

    # graph connectivity

    def remove_unconnected(self):
        valid_categories = {self.root_node} | nx.descendants(self.graph, self.root_node)
        self._remove_all_nodes_except(valid_categories)
        return self

    def append_unconnected(self):
        unconnected_root_nodes = {node for node in self.nodes if not self.parents(node) and node != self.root_node}
        self._add_edges([(self.root_node, node) for node in unconnected_root_nodes])
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
                    edges_to_remove.add(current_edge)
        self._remove_edges(edges_to_remove)

    # semantic connectivity

    def remove_unrelated_edges(self):
        valid_edges = set()
        node_to_headlemmas_mapping = {node: nlp_util.get_head_lemmas(nlp_util.parse(self.get_name(node))) for node in self.nodes}
        for parent, child in self.edges:
            parent_lemmas = node_to_headlemmas_mapping[parent]
            child_lemmas = node_to_headlemmas_mapping[child]
            if any(hypernymy_util.is_hypernym(pl, cl) for pl in parent_lemmas for cl in child_lemmas):
                valid_edges.add((parent, child))
        valid_nodes = {e[0] for e in valid_edges} | {e[1] for e in valid_edges} | {self.root_node}
        # remove all unrelated nodes
        self._remove_all_nodes_except(valid_nodes)
        # remove all unrelated edges
        self._remove_all_edges_except(valid_edges)
        return self

    # compound nodes

    def get_node_for_part(self, part: str) -> Optional[str]:
        if self._node_by_part is None:  # initialise part-to-node index if not existing
            self._node_by_part = {part: node for node in self.nodes for part in self.get_parts(node)}
        return self._node_by_part[part] if part in self._node_by_part else None

    def get_parts(self, node: str) -> set:
        if not self.has_node(node):
            raise Exception(f'Node {node} not in graph.')
        return self._get_attr(node, self.ATTRIBUTE_PARTS)

    def _set_parts(self, node: str, parts: set):
        if not self.has_node(node):
            raise Exception(f'Node {node} not in category graph.')
        self._set_attr(node, self.ATTRIBUTE_PARTS, parts)
        self._node_by_part = None  # reset part-to-node index due to changes

    def merge_nodes_alt(self):
        nodes_containing_by = {node for node in self.nodes if '_by_' in node}
        nodes_canonical_names = {}
        for node in nodes_containing_by:
            node_name = self.get_name(node)
            canonical_name = nlp_util.remove_by_phrase_from_text(node_name)
            if node_name != canonical_name:
                nodes_canonical_names[node] = canonical_name
        remaining_nodes_to_merge = set(nodes_canonical_names)
        util.get_logger().debug(f'Found {len(remaining_nodes_to_merge)} nodes to merge.')

        # 1) direct merge and synonym merge
        direct_merges = defaultdict(set)

        nodes_important_words = {node: nlp_util.without_stopwords(canonical_name) for node, canonical_name in nodes_canonical_names.items()}
        for node in remaining_nodes_to_merge:
            node_important_words = nodes_important_words[node]
            for parent in self.parents(node):
                if parent not in nodes_important_words:
                    nodes_important_words[parent] = nlp_util.without_stopwords(self.get_name(parent))
                parent_important_words = nodes_important_words[parent]

                if hypernymy_util.phrases_are_synonymous(node_important_words, parent_important_words):
                    direct_merges[node].add(parent)
        util.get_logger().debug(f'Found {len(direct_merges)} nodes to merge directly.')
        util.get_logger().debug(f'Examples of direct merges:')
        for node in random.sample(direct_merges, min(len(direct_merges), 50)):
            util.get_logger().debug(f'{node} -> {direct_merges[node]}')

        # 2) category set merge
        catset_merges = defaultdict(set)
        remaining_nodes_to_merge = remaining_nodes_to_merge.difference(set(direct_merges))
        for node in remaining_nodes_to_merge:
            node_canonical_name = nodes_canonical_names[node]
            for parent in self.parents(node):
                similar_children_count = len({child for child in self.children(parent) if child in nodes_canonical_names and nodes_canonical_names[child] == node_canonical_name})
                if similar_children_count > 1:
                    catset_merges[node].add(parent)
        util.get_logger().debug(f'Found {len(catset_merges)} nodes to merge via category sets.')
        util.get_logger().debug(f'Examples of catset merges:')
        for node in random.sample(catset_merges, min(len(catset_merges), 50)):
            util.get_logger().debug(f'{node} -> {catset_merges[node]}')

        remaining_nodes_to_merge = remaining_nodes_to_merge.difference(set(catset_merges))
        util.get_logger().debug(f'Examples of the {len(remaining_nodes_to_merge)} remaining nodes left:')
        for node in random.sample(remaining_nodes_to_merge, min(len(remaining_nodes_to_merge), 100)):
            util.get_logger().debug(f'{node}')

        # in caligraph: simply remove by phrase (maybe do that only in caligraph namespace then) -> !! and only do it if there is no uppercase word in by-phrase !!
        # (and make sure to check whether the pruned category name also exists as dbpedia category)

    def merge_nodes(self):
        """Create compounds of nodes by merging similar nodes into one."""
        found_merge_target = True
        traversed_nodes = set()
        while found_merge_target:
            found_merge_target = False
            for node, _ in nx.traversal.bfs_edges(self.graph, self.root_node):
                if node in traversed_nodes:
                    continue

                children_to_merge = {c for c in self.children(node) if self._should_merge_nodes(node, c)}
                if children_to_merge:
                    found_merge_target = True
                    merge_target = node
                    break
                else:
                    traversed_nodes.add(node)
            if found_merge_target:
                #util.get_logger().debug(f'Merging nodes "{children_to_merge}" into {merge_target}.')
                for child in children_to_merge:
                    parents = {parent for parent in self.parents(child)}
                    grandchildren = {grandchild for grandchild in self.children(child)}
                    edges_to_add = {(parent, grandchild) for parent in parents for grandchild in grandchildren}
                    self._remove_nodes({child})
                    self._add_edges(edges_to_add)

                node_parts = self.get_parts(merge_target) | children_to_merge
                self._set_parts(merge_target, node_parts)

        return self

    def _should_merge_nodes(self, parent: str, child: str) -> bool:
        return child.startswith(parent) and nlp_util.remove_by_phrase_from_text(self.get_name(child)) == self.get_name(parent)
