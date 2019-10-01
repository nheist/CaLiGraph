import networkx as nx
import util
from impl.util.hierarchy_graph import HierarchyGraph
import impl.util.rdf as rdf_util
import impl.category.base as cat_base
import impl.category.store as cat_store
import impl.category.util as cat_util
from impl.category.graph import CategoryGraph
import impl.list.base as list_base
import impl.list.util as list_util
from impl.list.graph import ListGraph
import impl.list.mapping as list_mapping
import impl.util.nlp as nlp_util
import numpy as np
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.dbpedia.heuristics as dbp_heur
import impl.caligraph.ontology_mapper as cali_mapping
import impl.caligraph.cali2ax as cali_axioms
import impl.caligraph.util as cali_util
from collections import defaultdict
from typing import Optional


class CaLiGraph(HierarchyGraph):

    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or rdf_util.CLASS_OWL_THING)
        self._node_dbpedia_types = defaultdict(set)
        self._node_resource_stats = defaultdict(dict)
        self._node_resources = defaultdict(set)
        self._resource_nodes = defaultdict(set)
        self._resource_provenance = defaultdict(set)
        self._node_axioms = defaultdict(set)
        self._node_axioms_transitive = defaultdict(set)

    def _reset_node_indices(self):
        self._node_dbpedia_types = defaultdict(set)
        self._node_resources = defaultdict(set)
        self._resource_nodes = defaultdict(set)
        self._resource_provenance = defaultdict(set)
        self._node_resource_stats = defaultdict(dict)

    def _reset_edge_indices(self):
        self._node_dbpedia_types = defaultdict(set)
        self._node_resources = defaultdict(set)
        self._resource_nodes = defaultdict(set)
        self._resource_provenance = defaultdict(set)
        self._node_resource_stats = defaultdict(dict)

    def get_label(self, item: str) -> Optional[str]:
        if cali_util.is_clg_type(item):
            return cali_util.clg_type2name(item)
        if cali_util.is_clg_resource(item):
            label = cali_util.clg_resource2name(item)
            label = nlp_util.remove_parentheses_content(label)
            return label.strip()
        return None

    def get_resources(self, node: str) -> set:
        if node not in self._node_resources:
            disjoint_types = {dt for t in self.get_dbpedia_types(node) for dt in dbp_heur.get_disjoint_types(t)}
            dbp_resources = self.get_dbpedia_resources(node)
            dbp_resources = {r for r in dbp_resources if not disjoint_types.intersection(dbp_store.get_types(r))}
            self._node_resources[node] = {cali_util.dbp_resource2clg_resource(r) for r in dbp_resources}
        return self._node_resources[node]

    def get_all_resources(self) -> set:
        return {r for n in self.nodes for r in self.get_resources(n)}

    def get_nodes_for_resource(self, resource: str) -> set:
        if not self._resource_nodes:
            for node in self.nodes:
                for r in self.get_resources(node):
                    self._resource_nodes[r].add(node)
        return self._resource_nodes[resource]

    def get_dbpedia_resources(self, node: str, use_listpage_resources=True) -> set:
        resources = set()
        for part in self.get_parts(node):
            if cat_util.is_category(part):
                resources.update({r for r in cat_store.get_resources(part) if dbp_store.is_possible_resource(r)})
            elif use_listpage_resources and list_util.is_listpage(part):
                resources.update({r for r in list_base.get_listpage_entities(self, part) if dbp_store.is_possible_resource(r)})
        return resources

    def get_resource_provenance(self, resource: str) -> set:
        if not self._resource_provenance:
            for node in self.nodes:
                for part in self.get_parts(node):
                    if cat_util.is_category(part):
                        for res in cat_store.get_resources(part):
                            self._resource_provenance[res].add(part)
                    elif list_util.is_listpage(part):
                        for res in list_base.get_listpage_entities(self, part):
                            self._resource_provenance[res].add(part)
        return self._resource_provenance[resource]

    def get_dbpedia_types(self, node: str, force_recompute=False) -> set:
        if node not in self._node_dbpedia_types or force_recompute:
            node_types = {p for p in self.get_parts(node) if dbp_util.is_dbp_type(p)}
            parent_types = {t for parent in self.parents(node) for t in self.get_dbpedia_types(parent)}
            self._node_dbpedia_types[node] = node_types | parent_types
        return self._node_dbpedia_types[node]

    def get_property_frequencies(self, node: str) -> dict:
        resource_stats = self.get_resource_stats(node)
        property_counts = resource_stats['property_counts']
        resource_count = resource_stats['resource_count']
        return {p: count / resource_count for p, count in property_counts.items()}

    def get_resource_stats(self, node: str) -> dict:
        if node not in self._node_resource_stats:
            resource_count = 0
            new_resource_count = 0
            property_counts = defaultdict(int)
            for res in self.get_dbpedia_resources(node, True):
                if res in dbp_store.get_resources():
                    resource_count += 1
                    for pred, values in dbp_store.get_properties(res).items():
                        for val in values:
                            property_counts[(pred, val)] += 1
                else:
                    new_resource_count += 1
            for child in self.children(node):
                child_stats = self.get_resource_stats(child)
                resource_count += child_stats['resource_count']
                for prop, count in child_stats['property_counts'].items():
                    property_counts[prop] += count
            self._node_resource_stats[node] = {'resource_count': resource_count, 'new_resource_count': new_resource_count, 'property_counts': property_counts}
        return self._node_resource_stats[node]

    def get_axioms(self, node: str, transitive=True) -> set:
        if node not in self._node_axioms_transitive:
            self._node_axioms_transitive[node] = self._node_axioms[node] | {ax for p in self.parents(node) for ax in self.get_axioms(p)}
        return self._node_axioms_transitive[node] if transitive else self._node_axioms[node]

    def get_all_properties(self):
        return {p for axioms in self._node_axioms.values() for p, _ in axioms}

    @property
    def statistics(self) -> str:
        leaf_nodes = {node for node in self.nodes if not self.children(node)}
        node_depths = self.depths()

        class_count = len(self.nodes)
        edge_count = len(self.edges)
        predicate_count = len({pred for axioms in self._node_axioms.values() for pred, _ in axioms})
        axiom_count = sum([len(axioms) for axioms in self._node_axioms.values()])
        parts_count = len({p for n in self.nodes for p in self.get_parts(n)})
        cat_parts_count = len({p for n in self.nodes for p in self.get_parts(n) if cat_util.is_category(p)})
        list_parts_count = len({p for n in self.nodes for p in self.get_parts(n) if list_util.is_listpage(p)})
        listcat_parts_count = len({p for n in self.nodes for p in self.get_parts(n) if list_util.is_listcategory(p)})
        classtree_depth_avg = np.mean([node_depths[node] for node in leaf_nodes])
        branching_factor_avg = np.mean([d for _, d in self.graph.out_degree if d > 0])

        category_instances = set()
        list_instances = set()

        for node in self.nodes:
            disjoint_types = {dt for t in self.get_dbpedia_types(node) for dt in dbp_heur.get_disjoint_types(t)}
            for part in self.get_parts(node):
                if cat_util.is_category(part):
                    cat_resources = {r for r in cat_store.get_resources(part) if dbp_store.is_possible_resource(r) and not disjoint_types.intersection(dbp_store.get_types(r))}
                    category_instances.update(cat_resources)
                elif list_util.is_listpage(part):
                    list_resources = {r for r in list_base.get_listpage_entities(self, part) if dbp_store.is_possible_resource(r) and not disjoint_types.intersection(dbp_store.get_types(r))}
                    list_instances.update(list_resources)

        return '\n'.join([
            '{:^40}'.format('STATISTICS'),
            '=' * 40,
            '{:<30} | {:>7}'.format('nodes', class_count),
            '{:<30} | {:>7}'.format('nodes below root', len(self.children(self.root_node))),
            '{:<30} | {:>7}'.format('edges', edge_count),
            '{:<30} | {:>7}'.format('predicates', predicate_count),
            '{:<30} | {:>7}'.format('axioms', axiom_count),
            '{:<30} | {:>7}'.format('parts', parts_count),
            '{:<30} | {:>7}'.format('category parts', cat_parts_count),
            '{:<30} | {:>7}'.format('list parts', list_parts_count),
            '{:<30} | {:>7}'.format('listcat parts', listcat_parts_count),
            '{:<30} | {:>7.2f}'.format('classtree depth', classtree_depth_avg),
            '{:<30} | {:>7.2f}'.format('branching factor', branching_factor_avg),
            '-' * 40,
            '{:<30} | {:>7}'.format('instances', len(category_instances | list_instances)),
            '{:<30} | {:>7}'.format('category instances', len(category_instances)),
            '{:<30} | {:>7}'.format('list instances', len(list_instances)),
            '{:<30} | {:>7}'.format('new instances', len(list_instances.difference(dbp_store.get_resources()))),
            ])

    @classmethod
    def build_graph(cls):
        util.get_logger().info('CaLiGraph: Starting to merge CategoryGraph and ListGraph..')
        graph = CaLiGraph(nx.DiGraph())

        # add root node
        graph._add_nodes({graph.root_node})
        graph._set_parts(graph.root_node, {graph.root_node, util.get_config('category.root_category')})

        cat_graph = cat_base.get_merged_graph()
        edge_count = len(cat_graph.edges)

        # initialise from category graph
        util.get_logger().debug('CaLiGraph: Starting CategoryMerge..')
        cat_node_names = {}
        for idx, node in enumerate(cat_graph.nodes):
            if idx % 10000 == 0:
                util.get_logger().debug(f'CaLiGraph: CategoryMerge - Created names for {idx} of {len(cat_graph.nodes)} nodes.')
            cat_node_names[node] = cls._get_canonical_name(cat_graph.get_name(node))

        for edge_idx, (parent_cat, child_cat) in enumerate(nx.bfs_edges(cat_graph.graph, cat_graph.root_node)):
            if edge_idx % 1000 == 0:
                util.get_logger().debug(f'CaLiGraph: CategoryMerge - Processed {edge_idx} of {edge_count} category edges.')

            parent_nodes = graph.get_nodes_for_part(parent_cat)
            if not parent_nodes:
                raise ValueError(f'"{parent_cat}" is not in graph despite of BFS!')

            child_nodes = graph.get_nodes_for_part(child_cat)
            if not child_nodes:
                child_nodes.add(graph._add_category_to_graph(child_cat, cat_node_names[child_cat], cat_graph))

            graph._add_edges({(pn, cn) for pn in parent_nodes for cn in child_nodes})

        # merge with list graph
        util.get_logger().debug('CaLiGraph: Starting ListMerge..')
        list_graph = list_base.get_merged_listgraph()
        edge_count = len(list_graph.edges)

        list_node_names = {}
        for idx, node in enumerate(list_graph.nodes):
            if idx % 10000 == 0:
                util.get_logger().debug(f'CaLiGraph: ListMerge - Created names for {idx} of {len(list_graph.nodes)} nodes.')
            list_node_names[node] = cls._get_canonical_name(list_graph.get_name(node), disable_normalization=True)

        for edge_idx, (parent_lst, child_lst) in enumerate(nx.bfs_edges(list_graph.graph, list_graph.root_node)):
            if edge_idx % 1000 == 0:
                util.get_logger().debug(f'CaLiGraph: ListMerge - Processed {edge_idx} of {edge_count} list edges.')

            parent_nodes = graph.get_nodes_for_part(parent_lst)
            if not parent_nodes:
                raise ValueError(f'"{parent_lst}" is not in graph despite of BFS!')

            child_nodes = graph.get_nodes_for_part(child_lst)
            if not child_nodes:
                child_nodes.add(graph._add_list_to_graph(child_lst, list_node_names[child_lst], list_graph))

            graph._add_edges({(pn, cn) for pn in parent_nodes for cn in child_nodes if pn != cn})

        # remove transitive edges to root
        edges_to_remove = set()
        for node in graph.nodes:
            parents = graph.parents(node)
            if len(parents) > 1 and graph.root_node in parents:
                edges_to_remove.add((graph.root_node, node))

        graph._remove_edges(edges_to_remove)
        util.get_logger().debug(f'CaLiGraph: PostProcessing - Removed {len(edges_to_remove)} transitive root edges.')

        # add Wikipedia categories to nodes that have an exact name match
        for node in graph.nodes:
            if node == graph.root_node:
                continue
            cat_name = cat_util.name2category(graph.get_name(node))
            if cat_name in cat_store.get_categories() and cat_name not in graph.get_parts(node):
                graph._set_parts(node, graph.get_parts(node) | {cat_name})

        return graph

    def _add_category_to_graph(self, category: str, category_name: str, cat_graph: CategoryGraph) -> str:
        node_id = self._convert_to_clg_type(category_name)
        node_parts = cat_graph.get_parts(category)
        if self.has_node(node_id):
            # extend existing node in graph
            node_parts.update(self.get_parts(node_id))
        else:
            # create new node in graph
            self._add_nodes({node_id})
            self._set_name(node_id, category_name)
        self._set_parts(node_id, node_parts)
        return node_id

    def _add_list_to_graph(self, lst: str, lst_name: str, list_graph: ListGraph) -> str:
        node_id = self._convert_to_clg_type(lst_name, disable_normalization=False)
        if not self.has_node(node_id):
            node_id = self._convert_to_clg_type(lst_name, disable_normalization=True)
        node_parts = list_graph.get_parts(lst)

        # check for equivalent mapping and existing node_id (if they map to more than one node -> log error)
        equivalent_nodes = {node for eq_cat in list_mapping.get_equivalent_categories(lst) for node in self.get_nodes_for_part(eq_cat)}
        if self.has_node(node_id):
            equivalent_nodes.add(node_id)
        if len(equivalent_nodes) > 1:
            util.get_logger().debug(f'CaLiGraph: ListMerge - For "{lst}" multiple equivalent nodes have been found: {equivalent_nodes}.')
            equivalent_nodes = {node_id} if node_id in equivalent_nodes else equivalent_nodes
        if equivalent_nodes:
            main_node_id = equivalent_nodes.pop()
            self._set_parts(main_node_id, self.get_parts(main_node_id) | node_parts)
            return main_node_id

        # check for parents to initialise under (parent mapping)
        self._add_nodes({node_id})
        self._set_name(node_id, lst_name)
        self._set_parts(node_id, node_parts)
        parent_nodes = {node for parent_cat in list_mapping.get_parent_categories(lst) for node in self.get_nodes_for_part(parent_cat)}
        self._add_edges({(pn, node_id) for pn in parent_nodes})

        return node_id

    @staticmethod
    def _get_canonical_name(name: str, disable_normalization=False) -> str:
        name = name[4:] if name.startswith('the ') else name
        return nlp_util.get_canonical_name(name, disable_normalization=disable_normalization)

    @classmethod
    def _convert_to_clg_type(cls, caligraph_name: str, disable_normalization=False) -> str:
        caligraph_name = nlp_util.singularize_phrase(nlp_util.parse(caligraph_name, disable_normalization=disable_normalization))
        return cali_util.name2clg_type(caligraph_name)

    def merge_ontology(self, use_listpage_resources: bool):
        # compute mapping from caligraph-nodes to dbpedia-types
        node_to_dbp_types_mapping = cali_mapping.find_mappings(self, use_listpage_resources)

        # add dbpedia types to caligraph
        type_graph = dbp_store._get_type_graph()
        for parent_type, child_type in nx.bfs_edges(type_graph, rdf_util.CLASS_OWL_THING):
            if parent_type not in dbp_store.get_main_equivalence_types() or child_type not in dbp_store.get_main_equivalence_types():
                continue  # only use main types for mapping and avoid equivalent classes without additional meaning
            parent_nodes = self.get_nodes_for_part(parent_type)
            if not parent_nodes:
                raise ValueError(f'"{parent_type}" is not in graph despite of BFS!')

            child_nodes = self.get_nodes_for_part(child_type)
            if not child_nodes:
                child_nodes.add(self._add_dbp_type_to_graph(child_type))

            self._add_edges({(pn, cn) for pn in parent_nodes for cn in child_nodes if pn != cn})

        # connect caligraph-nodes with dbpedia-types
        for node, dbp_types in node_to_dbp_types_mapping.items():
            parent_nodes = {n for t in dbp_types for n in self.get_nodes_for_part(t)}.difference({node})
            if parent_nodes:
                self._remove_edges({(self.root_node, node)})
                self._add_edges({(parent, node) for parent in parent_nodes})

        self.resolve_cycles().append_unconnected()
        cali_mapping.resolve_disjointnesses(self, use_listpage_resources)

        return self

    def _add_dbp_type_to_graph(self, dbp_type: str) -> str:
        name = dbp_store.get_label(dbp_type)
        node_id = cali_util.name2clg_type(name)
        node_parts = dbp_store.get_equivalent_types(dbp_type)
        if self.has_node(node_id):
            # extend existing node in graph
            node_parts.update(self.get_parts(node_id))
        else:
            # create new node in graph
            self._add_nodes({node_id})
            self._set_name(node_id, name)
        self._set_parts(node_id, node_parts)
        return node_id

    def compute_axioms(self):
        for node, axioms in cali_axioms.extract_axioms(self).items():
            for ax in axioms:
                prop = cali_util.dbp_type2clg_type(ax[1])
                val = cali_util.dbp_resource2clg_resource(ax[2]) if dbp_util.is_dbp_resource(ax[2]) else ax[2]
                self._node_axioms[node].add((prop, val))
        return self
