import networkx as nx
import numpy as np
import util
from impl.util.base_graph import BaseGraph
import impl.util.rdf as rdf_util
import impl.category.base as cat_base
import impl.category.store as cat_store
import impl.category.util as cat_util
import impl.category.nlp as cat_nlp
import impl.dbpedia.store as dbp_store
from collections import defaultdict


class CaLiGraph(BaseGraph):
    PROPERTY_DBP_TYPES = 'dbp_types'
    PROPERTY_CATEGORY = 'category'

    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node or rdf_util.CLASS_OWL_THING)

    def dbp_types(self, clg_type: str) -> set:
        return self._get_attr(clg_type, self.PROPERTY_DBP_TYPES) or set()

    @property
    def statistics(self) -> str:
        type_count = len(self.nodes)
        edge_count = len(self.edges)
        avg_indegree = np.mean([d for _, d in self.graph.in_degree])
        avg_outdegree = np.mean([d for _, d in self.graph.out_degree])

        avg_dbp_types = np.mean([len(self.dbp_types(t)) for t in self.nodes])

        return '\n'.join([
            '{:^40}'.format('CALIGRAPH STATISTICS'),
            '=' * 40,
            '{:<30} | {:>7}'.format('clg-types', type_count),
            '{:<30} | {:>7}'.format('edges', edge_count),
            '{:<30} | {:>7.2f}'.format('in-degree', avg_indegree),
            '{:<30} | {:>7.2f}'.format('out-degree', avg_outdegree),
            '{:<30} | {:>7.2f}'.format('avg linked dbp-types', avg_dbp_types)
        ])

    def new_dbp_types(self):
        new_dbp_types = defaultdict(set)
        for clg_type in self.nodes:
            for res in self._resources(clg_type):
                existing_types = dbp_store.get_transitive_types(res)
                new_resource_types = self.dbp_types(clg_type).difference(existing_types)
                for new_type in new_resource_types:
                    if all(dbp_store.get_cooccurrence_frequency(new_type, existing_type) > 0 for existing_type in existing_types):
                        new_dbp_types[res].update(new_resource_types)
        return new_dbp_types

    def _resources(self, clg_type):
        category = self._get_attr(clg_type, self.PROPERTY_CATEGORY)
        return cat_store.get_resources(category)

    @staticmethod
    def create_from_category_graph():
        # todo: use new taxonomic graph!
        catgraph = cat_base.get_dbp_typed_category_graph()
        typed_cats = {cat for cat in catgraph.nodes if catgraph.dbp_types(cat)}
        # create basic graph
        caligraph_types = [(_category_to_clg_type(cat), {CaLiGraph.PROPERTY_CATEGORY: cat, CaLiGraph.PROPERTY_DBP_TYPES: catgraph.dbp_types(cat)}) for cat in typed_cats]
        caligraph_edges = [(_category_to_clg_type(parent), _category_to_clg_type(child)) for parent, child in catgraph.graph.edges if catgraph.dbp_types(parent).intersection(catgraph.dbp_types(child))]
        cat_nlp.persist_cache()  # persisting created singularization-cache
        graph = nx.DiGraph()
        graph.add_nodes_from(caligraph_types)
        graph.add_edges_from(caligraph_edges)
        # add root node as well as edges to root
        graph.add_node(rdf_util.CLASS_OWL_THING)
        graph.add_edges_from({(rdf_util.CLASS_OWL_THING, clg_type) for clg_type in graph.nodes if not graph.predecessors(clg_type)})
        return CaLiGraph(graph)


def _category_to_clg_type(category: str) -> str:
    singularized_category_identifier = cat_util.remove_category_prefix(cat_nlp.singularize(category))
    return util.get_config('caligraph.namespace') + singularized_category_identifier
