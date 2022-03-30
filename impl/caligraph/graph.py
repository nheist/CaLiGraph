import networkx as nx
import utils
from utils import get_logger
from impl.util.base_graph import BaseGraph
from impl.util.hierarchy_graph import HierarchyGraph
from impl.util.rdf import RdfResource
from impl import category
from impl.category.graph import CategoryGraph
from impl import listpage
from impl.listpage.graph import ListGraph
import impl.listpage.mapping as list_mapping
import impl.util.nlp as nlp_util
import impl.util.hypernymy as hypernymy_util
import numpy as np
import impl.dbpedia.heuristics as dbp_heur
import impl.caligraph.ontology_mapper as clg_mapping
import impl.caligraph.cali2ax as clg_axioms
import impl.caligraph.util as clg_util
from polyleven import levenshtein
from collections import defaultdict
from typing import Optional, Set
from impl import listing
from impl.dbpedia.resource import DbpResource, DbpListpage, DbpResourceStore
from impl.dbpedia.ontology import DbpType, DbpOntologyStore
from impl.dbpedia.category import DbpCategory, DbpListCategory, DbpCategoryStore

# TODO: do not use full uri as graph identifier but name
class ClgType(RdfResource):
    pass


class ClgResource(RdfResource):
    pass


class CaLiGraph(HierarchyGraph):
    """A graph of categories and lists that is enriched with resources extract from list pages."""
    # initialisations
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or utils.get_config('caligraph.root_node'))
        self.dbr = DbpResourceStore.instance()

        self.use_listing_resources = False
        self._node_dbpedia_types = defaultdict(set)
        self._node_resource_stats = defaultdict(dict)
        self._node_resources = defaultdict(set)
        self._node_direct_cat_entities = defaultdict(set)
        self._node_listing_resources = defaultdict(set)
        self._all_node_resources = set()
        self._resource_nodes = defaultdict(set)
        self._resource_relations = set()
        self._resource_provenance = defaultdict(set)
        self._resource_altlabels = defaultdict(set)
        self._node_axioms = defaultdict(set)
        self._node_axioms_transitive = defaultdict(set)
        self._node_disjoint_dbp_types = defaultdict(set)
        self._node_disjoint_dbp_types_transitive = defaultdict(set)

    def _reset_node_indices(self):
        super()._reset_node_indices()
        self._node_dbpedia_types = defaultdict(set)
        self._node_resource_stats = defaultdict(dict)
        self._node_resources = defaultdict(set)
        self._node_direct_cat_entities = defaultdict(set)
        self._node_listing_resources = defaultdict(set)
        self._all_node_resources = set()
        self._resource_nodes = defaultdict(set)
        self._resource_relations = set()
        self._resource_provenance = defaultdict(set)
        self._resource_altlabels = defaultdict(set)
        self._node_axioms = defaultdict(set)
        self._node_axioms_transitive = defaultdict(set)
        self._node_disjoint_dbp_types = defaultdict(set)
        self._node_disjoint_dbp_types_transitive = defaultdict(set)

    def _reset_edge_indices(self):
        self._reset_node_indices()

    def enable_listing_resources(self):
        self.use_listing_resources = True
        # reset all resource-related indices
        self._node_resource_stats = defaultdict(dict)
        self._node_resources = defaultdict(set)
        self._all_node_resources = set()
        self._resource_nodes = defaultdict(set)
        self._resource_relations = set()
        self._resource_provenance = defaultdict(set)
        self._resource_altlabels = defaultdict(set)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove entries that should not be pickled
        del state['_node_dbpedia_types']
        del state['_node_axioms_transitive']
        del state['_node_disjoint_dbp_types']
        del state['_node_disjoint_dbp_types_transitive']
        del state['_resource_relations']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # initialize non-pickled entries
        self._node_dbpedia_types = defaultdict(set)
        self._node_axioms_transitive = defaultdict(set)
        self._node_disjoint_dbp_types = defaultdict(set)
        self._node_disjoint_dbp_types_transitive = defaultdict(set)
        self._resource_relations = set()

    def get_category_parts(self, node: str) -> Set[DbpCategory]:
        return {p for p in self.get_parts(node) if isinstance(p, DbpCategory)}

    def get_list_parts(self, node: str) -> Set[DbpListpage]:
        return {p for p in self.get_parts(node) if isinstance(p, DbpListpage)}

    def get_type_parts(self, node: str) -> Set[DbpType]:
        return {p for p in self.get_parts(node) if isinstance(p, DbpType)}

    def get_label(self, item: str) -> Optional[str]:
        """Return the label of a CaLiGraph type or resource."""
        # TODO: First try to get the respective DBP label?
        # TODO: never return None
        if clg_util.is_clg_type(item):
            return clg_util.clg_class2name(item)
        if clg_util.is_clg_resource(item):
            label = clg_util.clg_resource2name(item)
            label = nlp_util.remove_bracket_content(label)
            return label.strip()
        return None

    def get_altlabels(self, item: str) -> Set[str]:
        """Return alternative labels for a CaLiGraph resource."""
        if not self._resource_altlabels:
            # retrieve alternative labels from anchor texts of DBpedia
            for ent in self.dbr.get_entities():
                self._resource_altlabels[clg_util.dbp_resource2clg_resource(ent)].update(ent.get_surface_forms())
            if self.use_listing_resources:
                # retrieve alternative labels from new page entities
                for ent, ent_data in listing.get_page_entities(self).items():
                    altlabels = {l for origin_data in ent_data.values() for l in origin_data['labels']}
                    self._resource_altlabels[clg_util.name2clg_resource(ent)].update(altlabels)
        return self._resource_altlabels[item]

    def get_resources(self, node: str) -> Set[str]:
        """Return all resources of a node."""
        if not self._node_resources:
            # collect all resources for nodes
            for n in self.nodes:
                category_entities = self.get_dbp_entities_from_categories(n)
                dbpedia_resources = {r for t in self.get_type_parts(n) for r in self.dbr.get_resources_of_type(t)}
                self._node_resources[n] = {clg_util.dbp_resource2clg_resource(r) for r in category_entities | dbpedia_resources}
                if self.use_listing_resources:
                    self._node_resources[n].update(self.get_resources_from_listings(n))
            if self.use_listing_resources:
                # discard types of resources that conflict with its original dbpedia types
                for n in self._node_resources:
                    n_types = self.get_transitive_dbpedia_type_closure(n)
                    n_disjoint_types = {dt for t in n_types for dt in dbp_heur.get_all_disjoint_types(t)}
                    self._node_resources[n] = {r for r in self._node_resources[n] if not n_disjoint_types.intersection(self.dbr.get_resource_by_name(r).get_types())}
                # discard resources that still have conflicting dbpedia types
                resources_to_discard = set()
                dbptype_resources = defaultdict(set)
                for n, n_resources in self._node_resources.items():
                    for t in self.get_transitive_dbpedia_type_closure(n):
                        dbptype_resources[t].update(n_resources)
                processed_dbptypes = set()
                for t in set(dbptype_resources):
                    processed_dbptypes.add(t)
                    for dt in dbp_heur.get_all_disjoint_types(t).difference(processed_dbptypes):
                        resources_to_discard.update(dbptype_resources[t].intersection(dbptype_resources[dt]))
                for n in self._node_resources:
                    self._node_resources[n] = self._node_resources[n].difference(resources_to_discard)
            # make sure that we only return the most specific nodes
            for n in self.traverse_nodes_topdown():
                for p in self.parents(n):
                    self._node_resources[p] = self._node_resources[p].difference(self._node_resources[n])
        return self._node_resources[node]

    def get_all_resources(self) -> set:
        """Return all resources contained in the graph."""
        if not self._all_node_resources:
            node_resources = {r for n in self.nodes for r in self.get_resources(n)}
            axiom_resources = {ax[1] for n in self.nodes for ax in self.get_axioms(n, transitive=False) if clg_util.is_clg_resource(ax[1])}
            self._all_node_resources = {r for r in (node_resources | axiom_resources)}
        return self._all_node_resources

    def get_nodes_for_resource(self, resource: str) -> set:
        """Return all nodes the resource is contained in."""
        if not self._resource_nodes:
            for node in self.nodes:
                for r in self.get_resources(node):
                    self._resource_nodes[r].add(node)
        return self._resource_nodes[resource]

    def get_dbp_entities_from_categories(self, node: str) -> Set[DbpResource]:
        """Return all DBpedia entities directly associated with the node through Wikipedia categories."""
        if node not in self._node_direct_cat_entities:
            cat_entities = {e for cat in self.get_category_parts(node) for e in cat.get_entities()}
            cat_entities = {self.dbr.resolve_spelling_redirect(e) for e in cat_entities}
            self._node_direct_cat_entities[node] = cat_entities
        return set(self._node_direct_cat_entities[node])

    def get_resources_from_listings(self, node: str) -> set:
        if not self._node_listing_resources:
            for ent, ent_data in listing.get_page_entities(self).items():
                ent_nodes = {clg_util.name2clg_type(t) for origin_data in ent_data.values() for t in origin_data['types']}
                ent_nodes.update({n for origin in ent_data for n in self.get_nodes_for_part(self.dbr.get_resource_by_name(origin)) if self.dbr.has_resource_with_name(origin)})
                ent_uri = clg_util.name2clg_resource(ent)
                for n in ent_nodes:
                    self._node_listing_resources[n].add(ent_uri)
        return self._node_listing_resources[node]

    def get_resource_provenance(self, resource: str) -> set:
        """Return provenance information of a resource, i.e. which categories and lists have been used to extract it."""
        if not self._resource_provenance:
            for node in self.nodes:
                node_prov = self.get_category_parts(node) | self.get_list_parts(node)
                for res in self.get_resources(node):
                    self._resource_provenance[res].update(node_prov)
        return self._resource_provenance[resource]

    def get_transitive_dbpedia_type_closure(self, node: str, force_recompute=False) -> Set[DbpType]:
        """Return all mapped DBpedia types of a node."""
        if node not in self._node_dbpedia_types or force_recompute:
            parent_types = {t for parent in self.parents(node) for t in self.get_transitive_dbpedia_type_closure(parent, force_recompute)}
            self._node_dbpedia_types[node] = self.get_type_parts(node) | parent_types
        return self._node_dbpedia_types[node]

    def get_property_frequencies(self, node: str) -> dict:
        """Return property frequencies for a given node."""
        resource_stats = self.get_resource_stats(node)
        resource_count = resource_stats['resource_count']
        property_counts = resource_stats['property_counts']
        if resource_count < 5:
            resource_count = resource_stats['transitive_resource_count']
            property_counts = resource_stats['transitive_property_counts']

        return {p: count / resource_count for p, count in property_counts.items()}

    def get_resource_stats(self, node: str) -> dict:
        """Return resource stats of a node (i.e. resource count and property count)."""
        if node not in self._node_resource_stats:
            resource_count = 0
            property_counts = defaultdict(int)

            transitive_resource_count = 0
            transitive_property_counts = defaultdict(int)

            for res in self.get_dbp_entities_from_categories(node):
                resource_count += 1
                transitive_resource_count += 1
                for prop in res.get_properties(as_tuple=True).items():
                    property_counts[prop] += 1
                    transitive_property_counts[prop] += 1
            for child in self.children(node):
                child_stats = self.get_resource_stats(child)
                transitive_resource_count += child_stats['transitive_resource_count']
                for prop, count in child_stats['transitive_property_counts'].items():
                    transitive_property_counts[prop] += count
            self._node_resource_stats[node] = {
                'resource_count': resource_count,
                'property_counts': property_counts,
                'transitive_resource_count': transitive_resource_count,
                'transitive_property_counts': transitive_property_counts
            }
        return self._node_resource_stats[node]

    def get_axioms(self, node: str, transitive=True) -> set:
        """Return all axioms (from CaLi2Ax) for a given node."""
        if node not in self._node_axioms_transitive:
            self._node_axioms_transitive[node] = self._node_axioms[node] | {ax for p in self.parents(node) for ax in self.get_axioms(p)}
        return self._node_axioms_transitive[node] if transitive else self._node_axioms[node]

    def get_all_predicates(self) -> dict:
        """Return all predicates used in CaLiGraph. Predicate maps to True, if ObjectPredicate, and False otherwise."""
        return {r[1]: clg_util.is_clg_resource(r[2]) for r in self.get_all_relations()}

    def get_all_relations(self) -> set:
        """Return all relations in CaLiGraph."""
        if not self._resource_relations:
            # add relations from axioms
            self._resource_relations.update(self.get_relations_from_axioms())
            # add relations from listings
            if self.use_listing_resources:
                for ent, ent_data in listing.get_page_entities(self).items():
                    ent_uri = clg_util.name2clg_resource(ent)
                    for origin_data in ent_data.values():
                        # TODO: won't work, as relations are currently passed as dbp-indices
                        self._resource_relations.update((ent_uri, clg_util.name2clg_prop(p), clg_util.name2clg_resource(o)) for p, o in origin_data['out'])
                        self._resource_relations.update((clg_util.name2clg_resource(s), clg_util.name2clg_prop(p), ent_uri) for p, s in origin_data['in'])
        return self._resource_relations

    def get_relations_from_axioms(self) -> set:
        relations_from_axioms = set()
        for node in self.nodes:
            for pred, val in self.get_axioms(node):
                relations_from_axioms.update({(res, pred, val) for res in self.get_resources(node)})
        return relations_from_axioms

    def get_disjoint_dbp_types(self, node: str, transitive_closure=True) -> set:
        if node not in self._node_disjoint_dbp_types:  # fetch disjoint dbp types of node
            self._node_disjoint_dbp_types[node] = {dt for t in self.get_type_parts(node) for dt in dbp_heur.get_direct_disjoint_types(t)}
            self._node_disjoint_dbp_types_transitive[node] = {dt for t in self.get_transitive_dbpedia_type_closure(node) for dt in dbp_heur.get_direct_disjoint_types(t)}
        return self._node_disjoint_dbp_types_transitive[node] if transitive_closure else self._node_disjoint_dbp_types[node]

    def get_direct_disjoint_nodes(self, node: str) -> set:
        disjoint_types = self.get_disjoint_dbp_types(node, transitive_closure=False)
        direct_disjoint_types = {t for t in disjoint_types if not any(t in self.get_disjoint_dbp_types(p, transitive_closure=False) for p in self.parents(node))}
        direct_disjoint_nodes = {n for t in direct_disjoint_types for n in self.get_nodes_for_part(t)}
        return {n for n in direct_disjoint_nodes if not self.ancestors(n).intersection(direct_disjoint_nodes)}

    def get_all_disjoint_nodes(self, node: str) -> set:
        global __DISJOINTNESS_BY_DBP_TYPE__  # todo: change to graph attribute
        if '__DISJOINTNESS_BY_DBP_TYPE__' not in globals():
            __DISJOINTNESS_BY_DBP_TYPE__ = defaultdict(set)
            for n in self.nodes:
                for dt in self.get_disjoint_dbp_types(n, transitive_closure=True):
                    __DISJOINTNESS_BY_DBP_TYPE__[dt].add(n)

        types = self.get_transitive_dbpedia_type_closure(node)
        return {n for t in types for n in __DISJOINTNESS_BY_DBP_TYPE__[t]}

    @property
    def statistics(self) -> str:
        """Return statistics of CaLiGraph in a printable format."""
        leaf_nodes = {node for node in self.nodes if not self.children(node)}
        node_depths = self.depths()

        class_count = len(self.nodes)
        classes_connected_to_dbpedia_count = len({n for n in self.nodes if self.get_transitive_dbpedia_type_closure(n)})
        edge_count = len(self.edges)
        predicate_count = len(self.get_all_predicates())
        axiom_predicate_count = len({pred for axioms in self._node_axioms.values() for pred, _ in axioms})
        parts_count = len({p for n in self.nodes for p in self.get_parts(n)})
        cat_parts_count = len({p for n in self.nodes for p in self.get_category_parts(n)})
        list_parts_count = len({p for n in self.nodes for p in self.get_list_parts(n)})
        listcat_parts_count = len({p for n in self.nodes for p in self.get_parts(n) if isinstance(p, DbpListCategory)})
        classtree_depth_avg = np.mean([node_depths[node] for node in leaf_nodes])
        branching_factor_avg = np.mean([d for _, d in self.graph.out_degree if d > 0])
        axiom_count = sum([len(axioms) for axioms in self._node_axioms.values()])
        resource_axiom_count = len([ax for axioms in self._node_axioms.values() for ax in axioms if clg_util.is_clg_resource(ax[1])])
        literal_axiom_count = axiom_count - resource_axiom_count
        direct_node_axiom_count = len({n for n in self.nodes if self.get_axioms(n, transitive=False)})
        node_axiom_count = len({n for n in self.nodes if self.get_axioms(n, transitive=True)})

        resources = self.get_all_resources()
        types_per_resource = np.mean([len(self.get_nodes_for_resource(r) | {tt for t in self.get_nodes_for_resource(r) for tt in self.ancestors(t)}) for r in resources])
        relations = self.get_all_relations()
        resource_relation_count = len({r for r in relations if clg_util.is_clg_resource(r[2])})
        literal_relation_count = len(relations) - resource_relation_count
        in_degree = resource_relation_count / len(resources)
        out_degree = len(relations) / len(resources)

        return '\n'.join([
            '{:^40}'.format('STATISTICS'),
            '=' * 40,
            '{:<30} | {:>7}'.format('nodes', class_count),
            '{:<30} | {:>7}'.format('nodes below root', len(self.children(self.root_node))),
            '{:<30} | {:>7}'.format('nodes connected to DBpedia', classes_connected_to_dbpedia_count),
            '{:<30} | {:>7}'.format('edges', edge_count),
            '{:<30} | {:>7}'.format('predicates', predicate_count),
            '{:<30} | {:>7}'.format('axiom predicates', axiom_predicate_count),
            '{:<30} | {:>7}'.format('parts', parts_count),
            '{:<30} | {:>7}'.format('category parts', cat_parts_count),
            '{:<30} | {:>7}'.format('list parts', list_parts_count),
            '{:<30} | {:>7}'.format('listcat parts', listcat_parts_count),
            '{:<30} | {:>7.2f}'.format('classtree depth', classtree_depth_avg),
            '{:<30} | {:>7.2f}'.format('branching factor', branching_factor_avg),
            '{:<30} | {:>7}'.format('axioms', axiom_count),
            '{:<30} | {:>7}'.format('resource axioms', resource_axiom_count),
            '{:<30} | {:>7}'.format('literal axioms', literal_axiom_count),
            '{:<30} | {:>7}'.format('nodes with direct axiom', direct_node_axiom_count),
            '{:<30} | {:>7}'.format('nodes with axiom', node_axiom_count),
            '-' * 40,
            '{:<30} | {:>7}'.format('resources', len(resources)),
            '{:<30} | {:>7}'.format('types per resource', types_per_resource),
            '{:<30} | {:>7}'.format('relations', len(relations)),
            '{:<30} | {:>7}'.format('resource relations', resource_relation_count),
            '{:<30} | {:>7}'.format('literal relations', literal_relation_count),
            '{:<30} | {:>7}'.format('resource in-degree', in_degree),
            '{:<30} | {:>7}'.format('resource out-degree', out_degree),
            ])

    @classmethod
    def build_graph(cls):
        """Initialise the graph by merging the category graph and the list graph."""
        get_logger().info('Building base graph..')
        dbo = DbpOntologyStore.instance()
        dbc = DbpCategoryStore.instance()
        graph = CaLiGraph(nx.DiGraph())

        # add root node
        graph._add_nodes({graph.root_node})
        graph._set_parts(graph.root_node, {dbo.get_type_root(), dbc.get_category_root()})

        # initialise from category graph
        cat_graph = category.get_merged_graph()
        cat_node_names = {n: cls._get_canonical_name(cat_graph.get_name(n)) for n in cat_graph.nodes}
        for parent_cat_node, child_cat_node in cat_graph.traverse_edges_topdown():
            parent_nodes = {n for c in cat_graph.get_categories(parent_cat_node) for n in graph.get_nodes_for_part(c)}
            if not parent_nodes:
                raise ValueError(f'"{parent_cat_node}" is not in graph despite of BFS!')

            child_nodes = {n for c in cat_graph.get_categories(child_cat_node) for n in graph.get_nodes_for_part(c)}
            if not child_nodes:
                child_nodes.add(graph._add_category_to_graph(child_cat_node, cat_node_names[child_cat_node], cat_graph))

            graph._add_edges({(pn, cn) for pn in parent_nodes for cn in child_nodes})

        # merge with list graph
        list_graph = listpage.get_merged_listgraph()
        list_node_names = {n: cls._get_canonical_name(list_graph.get_name(n)) for n in list_graph.nodes}
        for parent_list_node, child_list_node in list_graph.traverse_edges_topdown():
            parent_nodes = {n for lst in list_graph.get_lists(parent_list_node) for n in graph.get_nodes_for_part(lst)}
            if not parent_nodes:
                raise ValueError(f'"{parent_list_node}" is not in graph despite of BFS!')

            child_nodes = {n for lst in list_graph.get_lists(child_list_node) for n in graph.get_nodes_for_part(lst)}
            if not child_nodes:
                child_nodes.add(graph._add_list_to_graph(child_list_node, list_node_names[child_list_node], list_graph, cat_graph))

            graph._add_edges({(pn, cn) for pn in parent_nodes for cn in child_nodes if pn != cn})

        # remove transitive edges to root
        edges_to_remove = set()
        for node in graph.nodes:
            parents = graph.parents(node)
            if len(parents) > 1 and graph.root_node in parents:
                edges_to_remove.add((graph.root_node, node))
        graph._remove_edges(edges_to_remove)

        # add Wikipedia categories to nodes that have an exact name match
        for node in graph.content_nodes:
            node_name = graph.get_name(node)
            if dbc.has_category_with_name(node_name):
                node_parts = graph.get_parts(node)
                cat = dbc.get_category_by_name(node_name)
                if cat not in node_parts:
                    graph._set_parts(node, node_parts | {cat})

        graph.append_unconnected()
        get_logger().info(f'Built base graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.')
        return graph

    def _add_category_to_graph(self, cat_node: str, cat_name: str, cat_graph: CategoryGraph) -> str:
        """Add a category as new node to the graph."""
        node_id = self._convert_to_clg_type(cat_name)
        node_parts = cat_graph.get_parts(cat_node)
        if self.has_node(node_id):
            # extend existing node in graph
            node_parts.update(self.get_parts(node_id))
        else:
            # create new node in graph
            self._add_nodes({node_id})
            self._set_name(node_id, cat_name)
        self._set_parts(node_id, node_parts)
        return node_id

    def _add_list_to_graph(self, list_node: str, name: str, list_graph: ListGraph, cat_graph: CategoryGraph) -> str:
        """Add a list as new node to the graph."""
        node_id = self._convert_to_clg_type(name)
        if not self.has_node(node_id):
            # If node_id is not in the graph, then we try synonyms with max. edit-distance of 2
            # e.g. to cover cases where the type is named 'Organisation' and the category 'Organization'
            for name_variation in hypernymy_util.get_variations(name):
                node_id_alternative = clg_util.name2clg_type(name_variation)
                if self.has_node(node_id_alternative):
                    node_id = node_id_alternative
                    break
        node_parts = list_graph.get_parts(list_node)

        # check for equivalent mapping and existing node_id (if they map to more than one node -> log error)
        equivalent_categories = {cat for cat_node in list_mapping.get_equivalent_category_nodes(list_node) for cat in cat_graph.get_categories(cat_node)}
        equivalent_nodes = {n for cat in equivalent_categories for n in self.get_nodes_for_part(cat)}
        if self.has_node(node_id):
            equivalent_nodes.add(node_id)
        if len(equivalent_nodes) > 1:
            get_logger().debug(f'Multiple equivalent nodes found for "{list_node}" during list merge: {equivalent_nodes}')
            equivalent_nodes = {node_id} if node_id in equivalent_nodes else equivalent_nodes
        if equivalent_nodes:
            main_node_id = sorted(equivalent_nodes, key=lambda x: levenshtein(x, node_id))[0]
            self._set_parts(main_node_id, self.get_parts(main_node_id) | node_parts)
            return main_node_id

        # check for parents to initialise under (parent mapping)
        self._add_nodes({node_id})
        self._set_name(node_id, name)
        self._set_parts(node_id, node_parts)
        parent_cats = {cat for parent_cat_node in list_mapping.get_parent_category_nodes(list_node) for cat in cat_graph.get_categories(parent_cat_node)}
        parent_nodes = {node for parent_cat in parent_cats for node in self.get_nodes_for_part(parent_cat)}
        self._add_edges({(pn, node_id) for pn in parent_nodes})

        return node_id

    @staticmethod
    def _get_canonical_name(name: str) -> str:
        """Convert a name into the canonical format (c.f. `get_canonical_name` in nlp_util)."""
        name = name[4:] if name.startswith('the ') else name
        return nlp_util.get_canonical_name(name)

    @classmethod
    def _convert_to_clg_type(cls, caligraph_name: str) -> str:
        """Convert a name into a CaLiGraph type URI."""
        caligraph_name = nlp_util.singularize_phrase(caligraph_name)
        return clg_util.name2clg_type(caligraph_name)

    def merge_ontology(self):
        """Combine the category-list-graph with the DBpedia ontology."""
        get_logger().info('Building graph with merged ontology..')
        dbo = DbpOntologyStore.instance()
        # remove edges from graph where clear type conflicts exist
        conflicting_edges = clg_mapping.find_conflicting_edges(self)
        self._remove_edges(conflicting_edges)
        self.append_unconnected()

        # compute mapping from caligraph-nodes to dbpedia-types
        node_to_dbp_types_mapping = clg_mapping.find_mappings(self)

        # add dbpedia types to caligraph
        type_graph = BaseGraph(dbo.type_graph, root_node=dbo.get_type_root())
        for parent_type, child_type in type_graph.traverse_edges_topdown():
            # make sure to always use the same of all the equivalent types of a type
            parent_type = dbo.get_main_equivalence_type(parent_type)
            child_type = dbo.get_main_equivalence_type(child_type)

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

        self.append_unconnected()

        # resolve potential disjointnesses in the graph and append remaining nodes
        clg_mapping.resolve_disjointnesses(self)
        self.append_unconnected(aggressive=False)

        get_logger().info(f'Built graph with merged ontology with {len(self.nodes)} nodes and {len(self.edges)} edges.')
        return self

    def _add_dbp_type_to_graph(self, dbp_type: DbpType) -> str:
        """Add a DBpedia type as node to the graph."""
        name = dbp_type.get_label()
        node_id = clg_util.name2clg_type(name)
        if not self.has_node(node_id):
            # If node_id is not in the graph, then we try synonyms with max. edit-distance of 2
            # e.g. to cover cases where the type is named 'Organisation' and the category 'Organization'
            for name_variation in hypernymy_util.get_variations(name):
                node_id_alternative = clg_util.name2clg_type(name_variation)
                if self.has_node(node_id_alternative):
                    node_id = node_id_alternative
                    break
        node_parts = DbpOntologyStore.instance().get_equivalents(dbp_type)
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
        """Compute axioms for all nodes in the graph."""
        get_logger().info('Computing Cat2Ax axioms for axiom graph..')
        for node, axioms in clg_axioms.extract_axioms(self).items():
            for ax in axioms:
                prop = clg_util.dbp_class2clg_class(ax[1])
                val = clg_util.dbp_resource2clg_resource(ax[2]) if isinstance(ax[2], DbpResource) else ax[2]
                self._node_axioms[node].add((prop, val))
        # filter out axioms that can be inferred from parents
        for node in self.traverse_nodes_bottomup():
            parent_axioms = {ax for p in self.parents(node) for ax in self.get_axioms(p)}
            self._node_axioms[node] = self._node_axioms[node].difference(parent_axioms)
        return self
