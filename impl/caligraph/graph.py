from typing import Set, Dict, Tuple, List, Optional, Union
from collections import defaultdict, Counter
import networkx as nx
from polyleven import levenshtein
import utils
from utils import get_logger
from impl.util.base_graph import BaseGraph
from impl.util.hierarchy_graph import HierarchyGraph
import impl.util.string as str_util
import impl.util.rdf as rdf_util
from impl import category
from impl.category.graph import CategoryGraph
from impl import listpage
from impl.listpage.graph import ListGraph
import impl.listpage.mapping as list_mapping
import impl.util.nlp as nlp_util
import impl.util.hypernymy as hypernymy_util
import impl.category.cat2ax as cat_axioms
import impl.dbpedia.heuristics as dbp_heur
from impl.dbpedia.resource import DbpEntity, DbpListpage, DbpResourceStore
from impl.dbpedia.ontology import DbpType, DbpOntologyStore
from impl.dbpedia.category import DbpCategory, DbpListCategory, DbpCategoryStore


class CaLiGraph(HierarchyGraph):
    """A graph of categories and lists that is enriched with resources extract from list pages."""
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or utils.get_config('caligraph.root_node'))
        self._stripped_nodes = {}
        self._node_dbpedia_types = defaultdict(set)
        self._node_direct_cat_entities = defaultdict(set)

    def add_node(self, node: str, parts=None, parents=None) -> str:
        self._add_nodes({node})
        self._set_label(node, rdf_util.name2label(node))
        self._set_parts(node, parts or set())
        if parents:
            self._add_edges({(pn, node) for pn in parents})
        self._stripped_nodes[str_util.normalize_separators(node).lower()] = node
        return node

    def _reset_node_indices(self):
        super()._reset_node_indices()
        self._node_dbpedia_types = defaultdict(set)
        self._node_direct_cat_entities = defaultdict(set)

    def _reset_edge_indices(self):
        self._reset_node_indices()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove entries that should not be pickled
        del state['_stripped_nodes']
        del state['_node_dbpedia_types']
        del state['_node_direct_cat_entities']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # initialize non-pickled entries
        self._stripped_nodes = {}
        self._node_dbpedia_types = defaultdict(set)
        self._node_direct_cat_entities = defaultdict(set)

    def get_category_parts(self, node: str) -> Set[DbpCategory]:
        return {p for p in self.get_parts(node) if isinstance(p, DbpCategory)}

    def get_list_parts(self, node: str) -> Set[DbpListpage]:
        return {p for p in self.get_parts(node) if isinstance(p, DbpListpage)}

    def get_type_parts(self, node: str) -> Set[DbpType]:
        return {p for p in self.get_parts(node) if isinstance(p, DbpType)}

    def find_node(self, potential_node: str) -> Optional[str]:
        stripped_potential_node = str_util.normalize_separators(potential_node).lower()
        return self._stripped_nodes[stripped_potential_node] if stripped_potential_node in self._stripped_nodes else None

    def find_node_via_synonyms(self, label: str) -> Optional[str]:
        # try synonyms with max.edit-distance of 2
        # e.g. to cover cases where the type is named 'Organisation' and the category 'Organization'
        for name_variation in hypernymy_util.get_variations(label):
            node = self.find_node(self._convert_label_to_clg_node(name_variation))
            if node is not None:
                return node
        return None

    def get_transitive_dbpedia_types(self, node: str, force_recompute=False) -> Set[DbpType]:
        """Return all mapped DBpedia types of a node."""
        if node not in self._node_dbpedia_types or force_recompute:
            parent_types = {t for parent in self.parents(node) for t in self.get_transitive_dbpedia_types(parent, force_recompute)}
            self._node_dbpedia_types[node] = self.get_type_parts(node) | parent_types
        return self._node_dbpedia_types[node]

    # CALIGRAPH INITIALIZATION

    @classmethod
    def build_graph(cls):
        """Initialise the graph by merging the category graph and the list graph."""
        get_logger().info('Building base graph..')
        dbo = DbpOntologyStore.instance()
        dbc = DbpCategoryStore.instance()
        graph = CaLiGraph(nx.DiGraph())

        # add root node
        graph.add_node(graph.root_node, parts={dbo.get_type_root(), dbc.get_category_root()})

        # initialise from category graph
        cat_graph = category.get_merged_graph()
        cat_node_labels = {n: nlp_util.get_canonical_label(cat_graph.get_label(n)) for n in cat_graph.nodes}
        for parent_cat_node, child_cat_node in cat_graph.traverse_edges_topdown():
            parent_nodes = graph.get_nodes_for_part(dbc.get_category_by_name(parent_cat_node))
            if not parent_nodes:
                raise ValueError(f'"{parent_cat_node}" is not in graph despite of BFS!')

            child_nodes = graph.get_nodes_for_part(dbc.get_category_by_name(child_cat_node))
            if not child_nodes:
                child_nodes.add(graph._add_category_to_graph(child_cat_node, cat_node_labels[child_cat_node], cat_graph))

            graph._add_edges({(pn, cn) for pn in parent_nodes for cn in child_nodes if pn != cn})

        # merge with list graph
        list_graph = listpage.get_merged_listgraph()
        list_node_labels = {n: nlp_util.get_canonical_label(list_graph.get_label(n)) for n in list_graph.nodes}
        for parent_list_node, child_list_node in list_graph.traverse_edges_topdown():
            parent_nodes = graph.get_nodes_for_part(graph._get_list_for_node(parent_list_node))
            if not parent_nodes:
                raise ValueError(f'"{parent_list_node}" is not in graph despite of BFS!')

            child_nodes = graph.get_nodes_for_part(graph._get_list_for_node(child_list_node))
            if not child_nodes:
                child_nodes.add(graph._add_list_to_graph(child_list_node, list_node_labels[child_list_node], list_graph, cat_graph))

            graph._add_edges({(pn, cn) for pn in parent_nodes for cn in child_nodes if pn != cn})

        # remove all transitive edges to root
        for node in graph.content_nodes:
            parents = graph.parents(node)
            if graph.root_node in parents and len(parents) > 1:
                graph._remove_edges({(graph.root_node, node)})

        # add Wikipedia categories to nodes that have an exact name match
        for node in graph.content_nodes:
            if dbc.has_category_with_name(node):
                cat = dbc.get_category_by_name(node)
                node_parts = graph.get_parts(node)
                if cat not in node_parts:
                    graph._set_parts(node, node_parts | {cat})

        graph.append_unconnected()
        get_logger().info(f'Built base graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.')
        return graph

    def _add_category_to_graph(self, cat_node: str, cat_node_label: str, cat_graph: CategoryGraph) -> str:
        """Add a category as new node to the graph."""
        potential_node = self._convert_label_to_clg_node(cat_node_label)
        node = self.find_node(potential_node)
        if not node:
            node = self.add_node(potential_node)

        node_parts = cat_graph.get_parts(cat_node)
        self._set_parts(node, self.get_parts(node) | node_parts)
        return node

    def _get_list_for_node(self, list_node: str) -> Union[DbpListpage, DbpListCategory]:
        dbc = DbpCategoryStore.instance()
        dbr = DbpResourceStore.instance()
        if dbc.has_category_with_name(list_node):  # listcategory
            return dbc.get_category_by_name(list_node)
        elif dbr.has_resource_with_name(list_node):  # listpage
            return dbr.get_resource_by_name(list_node)
        raise ValueError(f'Could not find listcategory or listpage for node "{list_node}".')

    def _add_list_to_graph(self, list_node: str, list_node_label: str, list_graph: ListGraph, cat_graph: CategoryGraph) -> str:
        """Add a list as new node to the graph."""
        potential_node = self._convert_label_to_clg_node(list_node_label)
        node = self.find_node(potential_node)
        if node is None:
            # find node via synonyms
            node = self.find_node_via_synonyms(list_node_label)
        if node is None:
            # find node via equivalent categories
            equivalent_categories = {cat for cat_node in list_mapping.get_equivalent_category_nodes(list_node) for cat in cat_graph.get_categories(cat_node)}
            equivalent_nodes = {n for cat in equivalent_categories for n in self.get_nodes_for_part(cat)}
            if equivalent_nodes:
                if len(equivalent_nodes) > 1:
                    get_logger().debug(f'Multiple equivalent nodes found for "{list_node}" during list merge: {equivalent_nodes}')
                node = sorted(equivalent_nodes, key=lambda x: levenshtein(x, potential_node))[0]
        if node is None:
            # create new node and add edges to parents via categories (if available)
            parent_cats = {cat for parent_cat_node in list_mapping.get_parent_category_nodes(list_node) for cat in cat_graph.get_categories(parent_cat_node)}
            parent_nodes = {node for parent_cat in parent_cats for node in self.get_nodes_for_part(parent_cat)}
            node = self.add_node(potential_node, parent_nodes)

        node_parts = list_graph.get_parts(list_node)
        self._set_parts(node, self.get_parts(node) | node_parts)
        return node

    def merge_ontology(self):
        """Combine the category-list-graph with the DBpedia ontology."""
        get_logger().info('Building graph with merged ontology..')
        dbo = DbpOntologyStore.instance()
        # remove edges from graph where clear type conflicts exist
        conflicting_edges = self._find_conflicting_edges()
        self._remove_edges(conflicting_edges)
        self.append_unconnected()

        # compute mapping from caligraph-nodes to dbpedia-types
        node_to_dbp_types_mapping = self._find_mappings()

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
        self._resolve_disjointnesses()
        self.append_unconnected(aggressive=False)

        get_logger().info(f'Built graph with merged ontology with {len(self.nodes)} nodes and {len(self.edges)} edges.')
        return self

    def _add_dbp_type_to_graph(self, dbp_type: DbpType) -> str:
        """Add a DBpedia type as node to the graph."""
        type_label = dbp_type.get_label()
        potential_node = self._convert_label_to_clg_node(type_label)
        node = self.find_node(potential_node)
        if node is None:
            # find node via synonyms
            node = self.find_node_via_synonyms(type_label)
        if node is None:
            # create new node
            node = self.add_node(potential_node)

        node_parts = DbpOntologyStore.instance().get_equivalents(dbp_type)
        self._set_parts(node, self.get_parts(node) | node_parts)
        return node

    @classmethod
    def _convert_label_to_clg_node(cls, label: str) -> str:
        """Convert a name into a CaLiGraph type URI."""
        label = nlp_util.singularize_phrase(label.strip())
        return str_util.capitalize(rdf_util.label2name(label))

    def _find_conflicting_edges(self) -> Set[Tuple[str, str]]:
        conflicting_edges = set()
        head_subject_lemmas = self.get_node_LHS()
        direct_mappings = {node: self._find_dbpedia_parents(node, True) for node in self.nodes}
        for node in self.traverse_nodes_topdown():
            for child in self.children(node):
                if head_subject_lemmas[node] == head_subject_lemmas[child]:
                    continue
                parent_disjoint_types = {dt for t in direct_mappings[node] for dt in dbp_heur.get_all_disjoint_types(t)}
                child_types = set(direct_mappings[child])
                if child_types.intersection(parent_disjoint_types):
                    conflicting_edges.add((node, child))
        get_logger().debug(f'Found {len(conflicting_edges)} conflicting edges to remove.')
        return conflicting_edges

    def _find_mappings(self) -> Dict[str, Set[DbpType]]:
        """Return mappings from nodes in `graph` to DBpedia types retrieved from axioms of the Cat2Ax approach."""
        mappings = {node: self._find_dbpedia_parents(node, False) for node in self.nodes}

        # apply complete transitivity to the graph in order to discover disjointnesses
        for node in self.traverse_nodes_topdown():
            for parent in self.parents(node):
                for t, score in mappings[parent].items():
                    mappings[node][t] = max(mappings[node][t], score)

        # resolve basic disjointnesses
        for node in self.traverse_nodes_topdown():
            coherent_type_sets = self._find_coherent_type_sets(mappings[node])
            if len(coherent_type_sets) <= 1:  # no disjoint sets
                continue

            coherent_type_sets = [(cs, max(cs.values())) for cs in coherent_type_sets]
            max_set_score = max([cs[1] for cs in coherent_type_sets])
            max_set_score_count = len([cs for cs in coherent_type_sets if cs[1] == max_set_score])
            if max_set_score_count > 1:  # no single superior set -> remove all type mappings
                types_to_remove = {t: mappings[node][t] for cs in coherent_type_sets for t in cs[0]}
            else:  # there is one superior set -> remove types from all sets except for superior set
                types_to_remove = {t: mappings[node][t] for cs in coherent_type_sets for t in cs[0] if cs[1] < max_set_score}
            self._remove_types_from_mapping(mappings, node, types_to_remove)

        # remove transitivity from the mappings and create sets of types
        for node in self.traverse_nodes_bottomup():
            parent_types = {t for p in self.parents(node) for t in mappings[p]}
            node_types = set(mappings[node]).difference(parent_types)
            mappings[node] = DbpOntologyStore.instance().get_independent_types(node_types)

        return mappings

    def _resolve_disjointnesses(self):
        """Resolve violations of disjointness axioms that are created through the mapping to DBpedia types."""
        dbo = DbpOntologyStore.instance()
        for node in self.traverse_nodes_topdown():
            parents = self.parents(node)
            coherent_type_sets = self._find_coherent_type_sets({t: 1 for t in self.get_transitive_dbpedia_types(node, force_recompute=True)})
            if len(coherent_type_sets) > 1:
                transitive_types = {tt for ts in coherent_type_sets for t in ts for tt in dbo.get_transitive_supertypes(t, include_self=True)}
                direct_types = {t for t in self._find_dbpedia_parents(node, False)}
                if not direct_types:
                    # compute direct types by finding the best matching type from lex score
                    lex_scores = self._compute_type_lexicalisation_scores(node)
                    types = [{t: lex_scores[t] for t in ts} for ts in coherent_type_sets]
                    types = [(ts, max(ts.values())) for ts in types]
                    best_type, score = max(types, key=lambda x: x[1])
                    direct_types = set() if score == 0 else set(best_type)
                # make sure that types induced by parts are integrated in direct types
                part_types = {t for t in self.get_parts(node) if isinstance(t, DbpType)}
                direct_types = (direct_types | part_types).difference({dt for t in part_types for dt in dbp_heur.get_all_disjoint_types(t)})
                direct_types = {tt for t in direct_types for tt in dbo.get_transitive_supertypes(t, include_self=True)}

                invalid_types = transitive_types.difference(direct_types)
                new_parents = {p for p in parents if not invalid_types.intersection(self.get_transitive_dbpedia_types(p))}
                self._remove_edges({(p, node) for p in parents.difference(new_parents)})
                if not new_parents and direct_types:
                    independent_types = dbo.get_independent_types(direct_types)
                    node_with_descendants = {node} | self.descendants(node)
                    new_parents = {p for t in independent_types for p in self.get_nodes_for_part(t) if p not in node_with_descendants}
                    self._add_edges({(p, node) for p in new_parents})

    def _find_dbpedia_parents(self, node: str, direct_resources_only: bool) -> Dict[DbpType, float]:
        """Retrieve DBpedia types that can be used as parents for `node` based on axioms discovered for it."""
        type_lexicalisation_scores = defaultdict(lambda: 0.2, self._compute_type_lexicalisation_scores(node))
        type_resource_scores = defaultdict(lambda: 0.0, self._compute_type_resource_scores(node, direct_resources_only))

        overall_scores = {t: type_lexicalisation_scores[t] * type_resource_scores[t] for t in type_resource_scores}
        max_score = max(overall_scores.values(), default=0)
        if max_score < utils.get_config('cali2ax.pattern_confidence'):
            return defaultdict(float)

        mapped_types = {t: score for t, score in overall_scores.items() if score >= max_score}
        result = defaultdict(float)
        for t, score in mapped_types.items():
            for tt in DbpOntologyStore.instance().get_transitive_supertypes(t, include_self=True):
                result[tt] = max(result[tt], score)

        result = defaultdict(float, {t: score for t, score in result.items() if not dbp_heur.get_all_disjoint_types(t).intersection(set(result))})
        return result

    def _compute_type_lexicalisation_scores(self, node: str) -> Dict[DbpType, float]:
        lexhead_subject_lemmas = nlp_util.get_lexhead_subjects(self.get_label(node))
        return cat_axioms._get_type_surface_scores(lexhead_subject_lemmas, lemmatize=False)

    def _compute_type_resource_scores(self, node: str, direct_resources_only: bool) -> Dict[DbpType, float]:
        node_entities = self._get_dbp_entities_from_categories(node)
        if not direct_resources_only or len([e for e in node_entities if e.get_types()]) < 5:
            node_entities.update({e for sn in self.descendants(node) for e in self._get_dbp_entities_from_categories(sn)})
        if len(node_entities) < 5:
            return {}  # better not return anything, if number of resources is too small
        type_counts = Counter()
        for ent in node_entities:
            for t in ent.get_transitive_types():
                type_counts[t] += 1
        return {t: count / len(node_entities) for t, count in type_counts.items()}

    def _get_dbp_entities_from_categories(self, node: str) -> Set[DbpEntity]:
        """Return all DBpedia entities directly associated with the node through Wikipedia categories."""
        if node not in self._node_direct_cat_entities:
            cat_entities = {e for cat in self.get_category_parts(node) for e in cat.get_entities()}
            cat_entities = {DbpResourceStore.instance().resolve_spelling_redirect(e) for e in cat_entities}
            self._node_direct_cat_entities[node] = cat_entities
        return set(self._node_direct_cat_entities[node])

    def _find_coherent_type_sets(self, dbp_types: Dict[DbpType, float]) -> List[Dict[DbpType, float]]:
        """Find biggest subset of types in `dbp_types` that does not violate any disjointness axioms."""
        coherent_sets = []
        disjoint_type_mapping = {t: set(dbp_types).intersection(dbp_heur.get_all_disjoint_types(t)) for t in dbp_types}
        for t, score in dbp_types.items():
            # add a type t to all coherent sets that do not contain disjoint types of t
            disjoint_types = disjoint_type_mapping[t]
            found_set = False
            for cs in coherent_sets:
                if not disjoint_types.intersection(set(cs)):
                    cs[t] = score
                    found_set = True
            # or create a new coherent set if t can not be added to any existing set
            if not found_set:
                coherent_sets.append({t: score})

        if len(coherent_sets) > 1:
            # check that a type is in all its valid sets (as types can be in multiple sets)
            for t, score in dbp_types.items():
                disjoint_types = disjoint_type_mapping[t]
                for cs in coherent_sets:
                    if not disjoint_types.intersection(set(cs)):
                        cs[t] = score

            # remove types that exist in all sets, as they are not disjoint with anything
            # finally, add them as individual set in case that they have the highest score
            all_set_types = {}
            for t in dbp_types:
                if all(t in cs for cs in coherent_sets):
                    for cs in coherent_sets:
                        all_set_types[t] = cs[t]
                        del cs[t]
            coherent_sets.append(all_set_types)
            coherent_sets = [cs for cs in coherent_sets if cs]  # remove possibly empty coherent sets

        return coherent_sets

    def _remove_types_from_mapping(self, mappings: Dict[str, Dict[DbpType, float]], node: str, types_to_remove: Dict[DbpType, float]):
        """Remove `types_to_remove` from a mapping for a node in order to resolve disjointnesses."""
        dbo = DbpOntologyStore.instance()
        types_to_remove = {tt: score for t, score in types_to_remove.items() for tt in dbo.get_transitive_subtypes(t, include_self=True)}

        node_closure = self.ancestors(node)
        node_closure.update({d for n in node_closure for d in self.descendants(n)})
        for n in node_closure:
            mappings[n] = {t: score for t, score in mappings[n].items() if t not in types_to_remove or score > types_to_remove[t]}
