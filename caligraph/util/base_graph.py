import networkx as nx


class BaseGraph:
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        self.graph = graph
        self.root_node = root_node

    def copy(self):
        return self.__class__(self.graph.copy(), root_node=self.root_node)

    @property
    def nodes(self) -> set:
        return set(self.graph.nodes)

    @property
    def edges(self) -> set:
        return set(self.graph.edges)

    def _predecessors(self, node: str) -> set:
        return set(self.graph.predecessors(node))

    def _successors(self, node: str) -> set:
        return set(self.graph.successors(node))

    def _depth(self, node: str) -> int:
        return nx.shortest_path_length(self.graph, source=self.root_node, target=node)

    def _get_attr(self, node, attr):
        return self.graph.nodes(data=attr)[node]

    def _set_attr(self, node, attr, val):
        self.graph.nodes[node][attr] = val
