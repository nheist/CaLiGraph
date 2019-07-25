import networkx as nx
from typing import Optional
from impl.util.hierarchy_graph import HierarchyGraph
import impl.dbpedia.store as dbp_store
import impl.category.store as cat_store
import impl.category.util as cat_util
import impl.list.util as list_util
import util


class ListGraph(HierarchyGraph):
    # initialisations
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or util.get_config('category.root_category'))

    # node lists
    def get_node_for_list(self, lst: str) -> Optional[str]:
        return self.get_node_for_part(lst)

    def get_lists(self, node: str) -> set:
        return self.get_parts(node)

    def _set_lists(self, node: str, lst: set):
        self._set_parts(node, lst)

    # GRAPH CREATION

    @classmethod
    def create_from_dbpedia(cls):
        # add nodes and edges for listcategories
        nodes = {cat for cat in cat_store.get_categories() if list_util.is_listcategory(cat)}
        edges = set()
        for listcat in nodes:
            listcat_children = {child for child in cat_store.get_children(listcat) if child in nodes}
            edges.update({(listcat, child) for child in listcat_children})

        # add nodes and edges for listpages
        for listcat in list(nodes):
            listpages = {dbp_store.resolve_redirect(page) for page in cat_store.get_resources(listcat) if list_util.is_listpage(page)}
            listpages = {lp for lp in listpages if list_util.is_listpage(lp)}  # filter out redirects on non-listpages
            nodes.update(listpages)
            edges.update({(listcat, listpage) for listpage in listpages})

        # initialise graph
        graph = nx.DiGraph(incoming_graph_data=list(edges))
        graph.add_nodes_from(list({n for n in nodes.difference(set(graph.nodes))}))
        list_graph = ListGraph(graph)

        for node in graph.nodes:
            list_graph._set_name(node, list_util.list2name(node))
            list_graph._set_parts(node, {node})

        # add root node
        graph.add_node(list_graph.root_node)
        list_graph._set_name(list_graph.root_node, cat_util.category2name(list_graph.root_node))
        list_graph._set_parts(list_graph.root_node, {list_graph.root_node})

        return list_graph
