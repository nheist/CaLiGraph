from typing import Set, List
from collections import defaultdict
import itertools
import networkx as nx
from impl.wikipedia import MentionId
from entity_linking.entity_disambiguation.data import CandidateAlignment, DataCorpus
from entity_linking.entity_disambiguation.matching.matcher import MatcherWithCandidates


class TopDownFusionMatcher(MatcherWithCandidates):
    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        ag = self._get_alignment_graph(eval_mode)
        valid_subgraphs = self._compute_valid_subgraphs(ag)
        ca = CandidateAlignment()
        for g in valid_subgraphs:
            mention_nodes = self._get_mention_nodes(g)
            for mention_pair in itertools.combinations(mention_nodes, 2):
                ca.add_candidate(mention_pair, 1)
            ent_nodes = self._get_entity_nodes(g)
            if ent_nodes:
                ent_node = ent_nodes.pop()
                for m_id in mention_nodes:
                    ca.add_candidate((m_id, ent_node), 1)
        return ca

    def _get_alignment_graph(self, eval_mode: str) -> nx.Graph:
        ag = nx.Graph()
        for (m_id, e_id), score in self.me_ca[eval_mode].get_me_candidates():
            ag.add_node(e_id, is_ent=True)
            ag.add_edge(e_id, m_id, weight=score)
        ag.add_weighted_edges_from([(u, v, score) for (u, v), score in self.mm_ca[eval_mode].get_mm_candidates()])
        return ag

    def _compute_valid_subgraphs(self, ag: nx.Graph) -> List[nx.Graph]:
        valid_subgraphs = []
        for nodes in nx.connected_components(ag):
            sg = ag.subgraph(nodes)
            if self._is_valid_graph(sg):
                valid_subgraphs.append(sg)
            else:
                valid_subgraphs.extend(self._split_into_valid_subgraphs(sg))
        return valid_subgraphs

    def _is_valid_graph(self, ag: nx.Graph) -> bool:
        return len(self._get_entity_nodes(ag)) <= 1

    @classmethod
    def _get_entity_nodes(cls, g: nx.Graph) -> Set[int]:
        return {node for node, is_ent in g.nodes(data='is_ent') if is_ent}

    @classmethod
    def _get_mention_nodes(cls, g: nx.Graph) -> Set[MentionId]:
        return {node for node, is_ent in g.nodes(data='is_ent') if not is_ent}

    def _split_into_valid_subgraphs(self, ag: nx.Graph) -> List[nx.Graph]:
        node_groups = defaultdict(set)
        weight_fct = lambda u, v, edge_attrs: 1 - edge_attrs['weight']  # use inversed weight
        for node, path in nx.multi_source_dijkstra_path(ag, self._get_entity_nodes(ag), weight=weight_fct).items():
            ent_node = path[0]
            node_groups[ent_node].add(node)
        return [ag.subgraph(nodes) for nodes in node_groups.values()]
