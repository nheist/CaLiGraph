from typing import Set, List
from collections import defaultdict
import itertools
import networkx as nx
from impl.wikipedia import MentionId
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.matcher import MatcherWithCandidates


class TopDownFusionMatcher(MatcherWithCandidates):
    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> Set[Pair]:
        mm_candidates, me_candidates = self.mm_candidates[eval_mode], self.me_candidates[eval_mode]
        ag = self._get_alignment_graph(mm_candidates, me_candidates)
        valid_subgraphs = self._compute_valid_subgraphs(ag)
        alignment = set()
        for g in valid_subgraphs:
            mention_nodes = self._get_mention_nodes(g)
            alignment.update({Pair(*sorted(mention_pair), 1) for mention_pair in itertools.combinations(mention_nodes, 2)})
            ent_nodes = self._get_entity_nodes(g)
            if ent_nodes:
                ent_node = ent_nodes.pop()
                alignment.update({Pair(m_id, ent_node, 1) for m_id in mention_nodes})
        return alignment

    def _get_alignment_graph(self, mm_candidates: Set[Pair], me_candidates: Set[Pair]) -> nx.Graph:
        ag = nx.Graph()
        ag.add_nodes_from([p.target for p in me_candidates], is_ent=True)  # first initialize entity nodes
        ag.add_weighted_edges_from(mm_candidates | me_candidates)  # then add all edges
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
