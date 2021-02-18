import networkx as nx
from typing import Set
import impl.category.util as cat_util
import impl.category.store as cat_store
from impl.util.hierarchy_graph import HierarchyGraph
import utils
import impl.util.nlp as nlp_util
import impl.category.cat2ax as cat_axioms


class CategoryGraph(HierarchyGraph):
    """A graph of categories retrieved from Wikipedia categories."""
    # initialisations
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or utils.get_config('category.root_category'))

    # node categories
    def get_all_categories(self) -> set:
        return {cat for node in self.nodes for cat in self.get_categories(node)}

    def get_categories(self, node: str) -> set:
        return self.get_parts(node)

    def _set_categories(self, node: str, categories: set):
        self._set_parts(node, categories)

    def get_nodes_for_category(self, category: str) -> set:
        return self.get_nodes_for_part(category)

    # node resources

    def get_resources(self, node: str) -> set:
        if not self.has_node(node):
            raise Exception(f'Node {node} not in category graph.')
        return {res for cat in self.get_categories(node) for res in cat_store.get_resources(cat)}

    def get_statistics(self, node: str) -> dict:
        if not self.has_node(node):
            raise Exception(f'Node {node} not in category graph.')
        return cat_store.get_statistics(node)

    # GRAPH CREATION

    @classmethod
    def create_from_dbpedia(cls):
        edges = [(cat, subcat) for cat in cat_store.get_categories() for subcat in cat_store.get_children(cat)]
        graph = CategoryGraph(nx.DiGraph(incoming_graph_data=edges))

        for node in graph.nodes:
            graph._set_name(node, cat_util.category2name(node))
            graph._set_parts(node, {node})

        return graph

    # filter for conceptual categories

    def make_conceptual(self):
        """Remove all nodes that are non-conceptual (i.e. that do not represent a class in a taxonomy)."""
        cat_names = [cat_store.get_label(cat) for cat in self.nodes]
        conceptual_categories = {cat for cat, has_plural_lexhead in zip(self.nodes, nlp_util.has_plural_lexhead_subjects(cat_names)) if has_plural_lexhead}
        # clearing the graph of any invalid nodes
        self._remove_all_nodes_except(conceptual_categories | {self.root_node})
        return self

    # CATEGORY AXIOMS

    def get_axiom_edges(self) -> Set[tuple]:
        """Return all edges that are loosely confirmed by axioms (i.e. most children share the same pattern)."""
        valid_axiom_edges = set()
        for parent in self.content_nodes:
            parent_axioms = cat_axioms.get_type_axioms(parent)
            children = tuple(self.children(parent))
            child_axioms = {c: {a for a in cat_axioms.get_type_axioms(c)} for c in children}
            consistent_child_axioms = len(children) > 2 and any(all(any(a.implies(x) for x in child_axioms[c]) for c in children) for a in child_axioms[children[0]])
            for c in children:
                if consistent_child_axioms or any(ca.implies(pa) for ca in child_axioms[c] for pa in parent_axioms):
                    valid_axiom_edges.add((parent, c))
        return valid_axiom_edges
