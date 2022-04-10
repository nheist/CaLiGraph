from typing import Set, Union
import networkx as nx
from impl.util.hierarchy_graph import HierarchyGraph
from utils import get_logger
from impl.dbpedia.category import DbpCategoryStore, DbpCategory, DbpListCategory
from impl.dbpedia.resource import DbpListpage, DbpResourceStore


class ListGraph(HierarchyGraph):
    """A graph of list categories and list pages retrieved directly from DBpedia resources."""

    # initialisations
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or DbpCategoryStore.instance().get_category_root().name)

    # node lists
    def get_all_lists(self) -> Set[Union[DbpCategory, DbpListpage]]:
        return {lst for node in self.nodes for lst in self.get_lists(node)}

    def get_lists(self, node: str) -> Set[Union[DbpCategory, DbpListpage]]:
        return self.get_parts(node)

    def _set_lists(self, node: str, lists: Set[Union[DbpCategory, DbpListpage]]):
        self._set_parts(node, lists)

    def get_nodes_for_list(self, lst: Union[DbpCategory, DbpListpage]) -> Set[str]:
        return self.get_nodes_for_part(lst)

    # GRAPH CREATION

    @classmethod
    def create_from_dbpedia(cls):
        """Initialise the graph by combining list categories with list pages."""
        get_logger().info('Building base list graph..')
        dbr = DbpResourceStore.instance()
        dbc = DbpCategoryStore.instance()

        listcats = dbc.get_listcategories()
        lists = listcats | dbr.get_listpages()
        # gather listcat -> listcat edges
        edges = {(p.name, c.name) for p in listcats for c in dbc.get_children(p, include_listcategories=True) if isinstance(c, DbpListCategory)}
        # gather listcat -> listpage edges
        for listcat in listcats:
            lp_children = {dbr.resolve_redirect(res) for res in listcat.get_resources()}
            lp_children = {res for res in lp_children if isinstance(res, DbpListpage)}
            edges.update({(listcat.name, lp.name) for lp in lp_children})

        # initialise graph
        graph = nx.DiGraph(incoming_graph_data=list(edges))
        graph.add_nodes_from({lst.name for lst in lists})  # make sure all nodes are in the graph
        list_graph = ListGraph(graph)

        for lst in lists:
            list_graph._set_label(lst.name, lst.get_label())
            list_graph._set_parts(lst.name, {lst})

        # add root node
        graph.add_node(list_graph.root_node)
        list_graph._set_label(list_graph.root_node, list_graph.root_node)
        list_graph._set_parts(list_graph.root_node, {dbc.get_category_root()})

        list_graph.append_unconnected()

        get_logger().info(f'Built base list graph with {len(list_graph.nodes)} nodes and {len(list_graph.edges)} edges.')
        return list_graph

    def remove_leaf_listcategories(self):
        leaf_listcats = {n for n in self.nodes if not self.children(n) and not any(isinstance(p, DbpListpage) for p in self.get_lists(n))}
        self._remove_nodes(leaf_listcats)
        return self
