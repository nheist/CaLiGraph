from typing import Set, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from entity_linking.entity_disambiguation.matching.matcher import MatcherWithCandidates
import utils


class FusionMatcher(MatcherWithCandidates, ABC):
    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> Set[Pair]:
        mm_candidates, me_candidates = self.mm_candidates[eval_mode], self.me_candidates[eval_mode]
        ag = self._get_alignment_graph(mm_candidates, me_candidates)
        edges_to_delete = self._compute_edges_to_delete(ag)
        utils.get_logger().debug(f'Found {len(edges_to_delete)} edges to delete.')
        return (mm_candidates | me_candidates).difference(edges_to_delete)

    def _get_alignment_graph(self, mm_candidates: Set[Pair], me_candidates: Set[Pair]) -> nx.Graph:
        ag = nx.Graph()
        ag.add_nodes_from([p.target for p in me_candidates], is_ent=True)  # first initialize entity nodes
        ag.add_weighted_edges_from(mm_candidates | me_candidates)  # then add all edges
        return ag

    def _find_invalid_subgraphs(self, ag: nx.Graph) -> List[nx.Graph]:
        subgraphs = []
        ents_per_subgraph = []
        for nodes in nx.connected_components(ag):
            sg = ag.subgraph(nodes)
            num_ents = len(self._get_entity_nodes(sg))
            if num_ents > 1:  # subgraph contains more than one entity -> should be avoided
                subgraphs.append(sg.copy())
                ents_per_subgraph.append(num_ents)
        if subgraphs:
            utils.get_logger().debug(f'Found {len(ents_per_subgraph)} invalid subgraphs with avg./med. of {np.mean(ents_per_subgraph)}/{np.median(ents_per_subgraph)} entities.')
        return subgraphs

    @classmethod
    def _get_entity_nodes(cls, g: nx.Graph) -> set:
        return {node for node, is_ent in g.nodes(data='is_ent') if is_ent}

    @abstractmethod
    def _compute_edges_to_delete(self, ag: nx.Graph) -> Set[Pair]:
        pass


class WeakestMentionMatcher(FusionMatcher):
    def _compute_edges_to_delete(self, ag: nx.Graph) -> Set[Pair]:
        edges_to_delete = set()
        for sg in self._find_invalid_subgraphs(ag):
            ent_nodes = self._get_entity_nodes(sg)
            mm_edges = [e for e in sg.edges(data='weight') if e[0] not in ent_nodes and e[1] not in ent_nodes]
            edge_to_delete = min(mm_edges, key=lambda x: x[2])
            edges_to_delete.add(Pair(*edge_to_delete))
            sg.remove_edge(edge_to_delete[0], edge_to_delete[1])
            edges_to_delete.update(self._compute_edges_to_delete(sg))
        return edges_to_delete


class WeakestEntityMatcher(FusionMatcher):
    def _compute_edges_to_delete(self, ag: nx.Graph) -> Set[Pair]:
        edges_to_delete = set()
        for sg in self._find_invalid_subgraphs(ag):
            ent_nodes = self._get_entity_nodes(sg)
            me_edges = [e for e in sg.edges(data='weight') if e[0] in ent_nodes or e[1] in ent_nodes]
            edge_to_delete = min(me_edges, key=lambda x: x[2])
            edges_to_delete.add(Pair(*edge_to_delete))
            sg.remove_edge(edge_to_delete[0], edge_to_delete[1])
            edges_to_delete.update(self._compute_edges_to_delete(sg))
        return edges_to_delete


class WeakestLinkMatcher(FusionMatcher):
    def _compute_edges_to_delete(self, ag: nx.Graph) -> Set[Pair]:
        edges_to_delete = set()
        for sg in self._find_invalid_subgraphs(ag):
            edge_to_delete = min(sg.edges(data='weight'), key=lambda x: x[2])
            edges_to_delete.add(Pair(*edge_to_delete))
            sg.remove_edge(edge_to_delete[0], edge_to_delete[1])
            edges_to_delete.update(self._compute_edges_to_delete(sg))
        return edges_to_delete


class PrecisionWeightedWeakestLinkMatcher(WeakestLinkMatcher):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        self.mm_weight = params['mm_weight']
        self.me_weight = params['me_weight']

    def _get_alignment_graph(self, mm_candidates: Set[Pair], me_candidates: Set[Pair]) -> nx.Graph:
        ag = nx.Graph()
        ag.add_nodes_from([p.target for p in me_candidates], is_ent=True)
        edges = [(s, t, conf * self.mm_weight) for s, t, conf in mm_candidates]
        edges += [(s, t, conf * self.me_weight) for s, t, conf in me_candidates]
        ag.add_weighted_edges_from(edges)
        ag.nodes()
        return ag
