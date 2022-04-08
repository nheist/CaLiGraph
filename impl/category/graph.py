import networkx as nx
from typing import Set, Tuple
from impl.util.hierarchy_graph import HierarchyGraph
from utils import get_logger
import impl.util.nlp as nlp_util
import impl.category.cat2ax as cat_axioms
from impl.category.cat2ax import TypeAxiom
from impl.dbpedia.category import DbpCategory, DbpCategoryStore
from impl.dbpedia.resource import DbpResource


class CategoryGraph(HierarchyGraph):
    """A graph of categories retrieved from Wikipedia categories."""
    # initialisations
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or DbpCategoryStore.instance().get_category_root().name)

    # node categories
    def get_all_categories(self) -> Set[DbpCategory]:
        return {cat for node in self.nodes for cat in self.get_categories(node)}

    def get_categories(self, node: str) -> Set[DbpCategory]:
        return self.get_parts(node)

    def _set_categories(self, node: str, categories: Set[DbpCategory]):
        self._set_parts(node, categories)

    def get_nodes_for_category(self, category: DbpCategory) -> Set[str]:
        return self.get_nodes_for_part(category)

    # node resources

    def get_resources(self, node: str) -> Set[DbpResource]:
        if not self.has_node(node):
            raise Exception(f'Node {node} not in category graph.')
        return {res for cat in self.get_categories(node) for res in cat.get_resources()}

    # GRAPH CREATION

    @classmethod
    def create_from_dbpedia(cls):
        get_logger().info('Building conceptual category graph..')
        dbc = DbpCategoryStore.instance()

        categories = dbc.get_categories()
        edges = [(cat.name, subcat.name) for cat in categories for subcat in dbc.get_children(cat)]
        graph = nx.DiGraph(incoming_graph_data=edges)
        graph.add_nodes_from({c.name for c in categories})  # make sure all categories are in the graph
        cat_graph = CategoryGraph(graph)

        for cat in categories:
            cat_graph._set_label(cat.name, cat.get_label())
            cat_graph._set_categories(cat.name, {cat})

        cat_graph.make_conceptual()

        get_logger().info(f'Built conceptual category graph with {len(cat_graph.nodes)} nodes and {len(cat_graph.edges)} edges.')
        return cat_graph

    # filter for conceptual categories

    def make_conceptual(self):
        """Remove all nodes that are non-conceptual (i.e. that do not represent a class in a taxonomy)."""
        nodes = list(self.nodes)
        node_labels = [self.get_label(n) for n in nodes]
        conceptual_category_nodes = {n for n, plh in zip(nodes, nlp_util.has_plural_lexhead_subjects(node_labels)) if plh}

        # clearing the graph of any invalid nodes
        self._remove_all_nodes_except(conceptual_category_nodes)
        self.append_unconnected()
        return self

    # CATEGORY AXIOMS

    def get_axiom_edges(self) -> Set[Tuple[str, str]]:
        """Return all edges that are loosely confirmed by axioms (i.e. most children share the same pattern)."""
        valid_axiom_edges = set()
        for parent in self.content_nodes:
            parent_axioms = self._get_type_axioms_for_node(parent)
            children = tuple(self.children(parent))
            child_axioms = {c: self._get_type_axioms_for_node(c) for c in children}
            consistent_child_axioms = len(children) > 2 and any(all(any(a.implies(x) for x in child_axioms[c]) for c in children) for a in child_axioms[children[0]])
            for c in children:
                if consistent_child_axioms or any(ca.implies(pa) for ca in child_axioms[c] for pa in parent_axioms):
                    valid_axiom_edges.add((parent, c))
        return valid_axiom_edges

    def _get_type_axioms_for_node(self, node: str) -> Set[TypeAxiom]:
        return {a for c in self.get_categories(node) for a in cat_axioms.get_type_axioms(c)}
