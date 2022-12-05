from typing import Tuple, List, Union, Optional, Set, Iterable
import networkx as nx
import utils
from impl.wikipedia import MentionId
from impl.subject_entity.entity_disambiguation.data import CandidateAlignment, DataCorpus
from impl.subject_entity.entity_disambiguation.matching.util import MatchingScenario
from impl.subject_entity.entity_disambiguation.matching.matcher import MatcherWithCandidates


class BottomUpClusteringMatcher(MatcherWithCandidates):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.mm_threshold = params['mm_threshold']
        self.me_threshold = params['me_threshold']

    def _get_param_dict(self) -> dict:
        return super()._get_param_dict() | {'mmt': self.mm_threshold, 'met': self.me_threshold}

    def _train_model(self, train_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        ag, edges = self._get_base_graph_and_edges(eval_mode)
        for u, v in edges:
            ag.add_edge(u, v)
            if self._has_invalid_subgraph(ag, u):
                ag.remove_edge(u, v)
        clusters = [(self._get_mention_nodes(sg), self._get_entity_node(sg)) for sg in self._get_subgraphs(ag)]
        return CandidateAlignment(clusters)

    def _get_base_graph_and_edges(self, eval_mode: str) -> Tuple[nx.Graph, List[Tuple[MentionId, Union[MentionId, int]]]]:
        utils.get_logger().debug('Initializing alignment graph..')
        ag = nx.Graph()
        edges = []
        for (m_id, e_id), score in self.me_ca[eval_mode].get_me_candidates(True):
            if score > self.me_threshold:
                ag.add_node(m_id, is_ent=False)
                ag.add_node(e_id, is_ent=True)
                edges.append((m_id, e_id, score))
        for (m_one, m_two), score in self.mm_ca[eval_mode].get_mm_candidates(True):
            if score > self.mm_threshold:
                ag.add_node(m_one, is_ent=False)
                ag.add_node(m_two, is_ent=False)
                edges.append((m_one, m_two, score))
        ordered_edges = [(u, v) for u, v, _ in sorted(edges, key=lambda x: x[2], reverse=True)]
        return ag, ordered_edges

    def _has_invalid_subgraph(self, g: nx.Graph, node: MentionId) -> bool:
        node_subgraph = g.subgraph(nx.node_connected_component(g, node))
        return self._is_valid_graph(node_subgraph)

    def _is_valid_graph(self, ag: nx.Graph) -> bool:
        return len(self._get_entity_nodes(ag)) <= 1

    @classmethod
    def _get_subgraphs(cls, g: nx.Graph) -> Iterable[nx.Graph]:
        for nodes in nx.connected_components(g):
            yield g.subgraph(nodes)

    @classmethod
    def _get_mention_nodes(cls, g: nx.Graph) -> Set[MentionId]:
        return {node for node, is_ent in g.nodes(data='is_ent') if not is_ent}

    @classmethod
    def _get_entity_nodes(cls, g: nx.Graph) -> List[int]:
        return [node for node, is_ent in g.nodes(data='is_ent') if is_ent]

    @classmethod
    def _get_entity_node(cls, g: nx.Graph) -> Optional[int]:
        ent_nodes = cls._get_entity_nodes(g)
        return ent_nodes[0] if ent_nodes else None
