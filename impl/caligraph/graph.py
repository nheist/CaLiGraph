import networkx as nx
import util
from impl.util.hierarchy_graph import HierarchyGraph
import impl.util.rdf as rdf_util
import impl.category.base as cat_base
import impl.list.base as list_base
import impl.list.mapping as list_mapping
import impl.util.nlp as nlp_util


class CaLiGraph(HierarchyGraph):

    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or rdf_util.CLASS_OWL_THING)

    @classmethod
    def build_graph(cls):
        util.get_logger().info('CaLiGraph: Starting to merge CategoryGraph and ListGraph..')
        graph = CaLiGraph(nx.DiGraph())
        #graph_nodes = set()

        # add root node
        graph._add_nodes({graph.root_node})
        #graph_nodes.add(graph.root_node)
        graph._set_parts(graph.root_node, {graph.root_node, util.get_config('category.root_category')})

        cat_graph = cat_base.get_merged_graph()
        edge_count = len(cat_graph.edges)

        # initialise from category graph
        util.get_logger().debug('CaLiGraph: Starting CategoryMerge..')
        #cat_node_names = {}
        #for idx, node in enumerate(cat_graph.nodes):
        #    if idx % 10000 == 0:
        #        util.get_logger().debug(f'CaLiGraph: CategoryMerge - Created names for {idx} of {len(cat_graph.nodes)} nodes.')
        #    cat_node_names[node] = cls.get_caligraph_name(cat_graph.get_name(node))

        for edge_idx, (parent_cat, child_cat) in enumerate(nx.bfs_edges(cat_graph.graph, cat_graph.root_node)):
            if edge_idx % 1000 == 0:
                util.get_logger().debug(f'CaLiGraph: CategoryMerge - Processed {edge_idx} of {edge_count} category edges.')

            parent_nodes = graph.get_nodes_for_part(parent_cat)
            if not parent_nodes:
                raise ValueError(f'"{parent_cat}" is not in graph despite of BFS!')

            child_nodes = graph.get_nodes_for_part(child_cat)
            if not child_nodes:
                # initialise child_node in caligraph
                #node_name = cat_node_names[child_cat]
                node_name = cls.get_caligraph_name(cat_graph.get_name(child_cat))
                node_id = cls.get_caligraph_resource(node_name)
                node_parts = cat_graph.get_parts(child_cat)
                if graph.has_node(node_id):
                    # resolve conflicts with existing node
                    #existing_node_categories = graph.get_parts(node_id)
                    #if existing_node_categories.intersection(cat_graph.children(parent_cat)):
                    graph._set_parts(node_id, graph.get_parts(node_id) | node_parts)
                    child_nodes = {node_id}
                    #else:
                    #    util.get_logger().debug(f'CaLiGraph: CategoryMerge - Failed to include node "{child_cat}".')
                else:
                    # create new node in graph
                    graph._add_nodes({node_id})
                    #graph_nodes.add(node_id)
                    graph._set_name(node_id, node_name)
                    graph._set_parts(node_id, node_parts)
                    child_nodes = {node_id}

            if child_nodes:
                graph._add_edges({(pn, cn) for pn in parent_nodes for cn in child_nodes})

        # merge with list graph
        util.get_logger().debug('CaLiGraph: Starting ListMerge..')
        list_graph = list_base.get_merged_listgraph()
        edge_count = len(list_graph.edges)

        list_node_names = {}
        for idx, node in enumerate(list_graph.nodes):
            if idx % 1000 == 0:
                util.get_logger().debug(f'CaLiGraph: ListMerge - Created names for {idx} of {len(cat_graph.nodes)} nodes.')
            list_node_names[node] = cls.get_caligraph_name(list_graph.get_name(node))

        for edge_idx, (parent_lst, child_lst) in enumerate(nx.bfs_edges(list_graph.graph, list_graph.root_node)):
            if edge_idx % 1000 == 0:
                util.get_logger().debug(f'CaLiGraph: ListMerge - Processed {edge_idx} of {edge_count} list edges.')

            parent_nodes = graph.get_nodes_for_part(parent_lst)
            if not parent_nodes:
                # initialise parent_node in caligraph
                node_name = list_node_names[parent_lst]
                node_id = cls.get_caligraph_resource(node_name)
                node_parts = list_graph.get_parts(parent_lst)

                equivalent_categories = list_mapping.get_equivalent_categories(parent_lst)
                if equivalent_categories:
                    parent_nodes = {node for cat in equivalent_categories for node in graph.get_nodes_for_part(cat)}
                    for pn in parent_nodes:
                        graph._set_parts(pn, graph.get_parts(pn) | node_parts)
                else:
                    parent_categories = list_mapping.get_parent_categories(parent_lst)
                    if parent_categories:
                        grandparent_nodes = {node for cat in parent_categories for node in graph.get_nodes_for_part(cat)} or {graph.root_node}
                        if node_id in graph.nodes:
                            if node_id in ({grandparent_nodes} | {c for gpn in grandparent_nodes for c in graph.children(gpn)}):
                                graph._set_parts(node_id, graph.get_parts(node_id) | node_parts)
                                parent_nodes = {node_id}
                            else:
                                util.get_logger().debug(f'CaLiGraph: ListMerge - Failed to include parent list "{parent_lst}".')
                        else:
                            # create new node in graph
                            graph._add_nodes({node_id})
                            graph._set_name(node_id, node_name)
                            graph._set_parts(node_id, node_parts)
                            graph._add_edges((gpn, node_id) for gpn in grandparent_nodes)
                            parent_nodes = {node_id}

            child_nodes = graph.get_nodes_for_part(child_lst)
            if not child_nodes:
                # initialise child_node in caligraph
                node_name = list_node_names[child_lst]
                node_id = cls.get_caligraph_resource(node_name)
                node_parts = list_graph.get_parts(child_lst)

                equivalent_categories = list_mapping.get_equivalent_categories(child_lst)
                if equivalent_categories:
                    child_nodes = {node for cat in equivalent_categories for node in graph.get_nodes_for_part(cat)}
                    for cn in child_nodes:
                        graph._set_parts(cn, graph.get_parts(cn) | node_parts)
                else:
                    # todo: maybe simplify this part by generalizing it similar to parent-list merge
                    parent_categories = list_mapping.get_parent_categories(parent_lst)
                    if parent_categories:
                        otherparent_nodes = {node for cat in parent_categories for node in graph.get_nodes_for_part(cat)}
                        if node_id in graph.nodes:
                            if node_id in ({otherparent_nodes} | {c for gpn in otherparent_nodes for c in graph.children(gpn)}):
                                graph._set_parts(node_id, graph.get_parts(node_id) | node_parts)
                                child_nodes = {node_id}
                            else:
                                util.get_logger().debug(f'CaLiGraph: ListMerge - Failed to include child list "{child_lst}".')
                        else:
                            # create new node in graph with links to other parents
                            graph._add_nodes({node_id})
                            graph._set_name(node_id, node_name)
                            graph._set_parts(node_id, node_parts)
                            graph._add_edges((gpn, node_id) for gpn in otherparent_nodes)
                            child_nodes = {node_id}
                    else:
                        if node_id in graph.nodes:
                            if node_id in ({parent_nodes} | {c for gpn in parent_nodes for c in graph.children(gpn)}):
                                graph._set_parts(node_id, graph.get_parts(node_id) | node_parts)
                                child_nodes = {node_id}
                            else:
                                util.get_logger().debug(f'CaLiGraph: ListMerge - Failed to include child list "{child_lst}".')
                        else:
                            # create new node in graph
                            graph._add_nodes({node_id})
                            graph._set_name(node_id, node_name)
                            graph._set_parts(node_id, node_parts)
                            child_nodes = {node_id}

            if parent_nodes and child_nodes:
                graph._add_edges({(pn, cn) for pn in parent_nodes for cn in child_nodes if pn != cn})

        # clean up
        # todo: cleanup
        #   - append unconnected ?
        #   - check for cycles ?
        #   - remove transitive edges ?
        return graph

    @staticmethod
    def get_ontology_namespace():
        return util.get_config('caligraph.namespace.ontology')

    @staticmethod
    def get_resource_namespace():
        return util.get_config('caligraph.namespace.resource')

    @staticmethod
    def get_caligraph_name(name: str) -> str:
        name = name[4:] if name.startswith('the ') else name
        return nlp_util.get_canonical_name(name, strict_by_removal=False).capitalize()

    @classmethod
    def get_caligraph_resource(cls, caligraph_name: str) -> str:
        return cls.get_resource_namespace() + caligraph_name.replace(' ', '_')
