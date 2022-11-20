from typing import Set, List, Iterable, Tuple, Optional, Dict
from abc import ABC
from collections import defaultdict, Counter
import itertools
import networkx as nx
import utils
from impl.wikipedia import MentionId
from entity_linking.data import CandidateAlignment, DataCorpus
from entity_linking.matching.util import MatchingScenario
from entity_linking.matching.matcher import MatcherWithCandidates


class GreedyClusteringMatcher(MatcherWithCandidates, ABC):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.mm_threshold = params['mm_threshold']
        self.me_threshold = params['me_threshold']

    def _get_param_dict(self) -> dict:
        return super()._get_param_dict() | {'mmt': self.mm_threshold, 'met': self.me_threshold}

    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass  # no training necessary

    def _get_alignment_graph(self, eval_mode: str, add_entities: bool) -> nx.Graph:
        utils.get_logger().debug('Initializing alignment graph..')
        ag = nx.Graph()
        if add_entities:
            for (m_id, e_id), score in self.me_ca[eval_mode].get_me_candidates(True):
                if score <= self.me_threshold:
                    continue
                ag.add_node(e_id, is_ent=True)
                ag.add_edge(e_id, m_id, weight=max(0, 1-score))
        ag.add_weighted_edges_from([(u, v, max(0, 1-score)) for (u, v), score in self.mm_ca[eval_mode].get_mm_candidates(True) if score > self.mm_threshold])
        return ag

    def _get_subgraphs(self, ag: nx.Graph) -> Iterable[nx.Graph]:
        for nodes in nx.connected_components(ag):
            yield ag.subgraph(nodes)

    def _create_alignment(self, clusters: List[Tuple[Set[MentionId], Optional[int]]]) -> CandidateAlignment:
        ca = CandidateAlignment()
        for mentions, ent_id in clusters:
            for mention_pair in itertools.combinations(mentions, 2):
                ca.add_candidate(mention_pair, 1)
            if ent_id is not None:
                for m_id in mentions:
                    ca.add_candidate((m_id, ent_id), 1)
        ca.add_entity_clustering([mentions for mentions, _ in clusters])
        return ca

    @classmethod
    def _get_mention_nodes(cls, g: nx.Graph) -> Set[MentionId]:
        return {node for node, is_ent in g.nodes(data='is_ent') if not is_ent}


class NastyLinker(GreedyClusteringMatcher):
    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        ag = self._get_alignment_graph(eval_mode, True)
        valid_subgraphs = self._compute_valid_subgraphs(ag)
        clusters = [(self._get_mention_nodes(g), self._get_entity_node(g)) for g in valid_subgraphs]
        return self._create_alignment(clusters)

    def _compute_valid_subgraphs(self, ag: nx.Graph) -> List[nx.Graph]:
        utils.get_logger().debug('Computing valid subgraphs..')
        valid_subgraphs = []
        for sg in self._get_subgraphs(ag):
            if self._is_valid_graph(sg):
                valid_subgraphs.append(sg)
            else:
                valid_subgraphs.extend(self._split_into_valid_subgraphs(sg))
        return valid_subgraphs

    def _is_valid_graph(self, ag: nx.Graph) -> bool:
        return len(self._get_entity_nodes(ag)) <= 1

    @classmethod
    def _get_entity_node(cls, g: nx.Graph) -> Optional[int]:
        ent_nodes = cls._get_entity_nodes(g)
        return ent_nodes[0] if ent_nodes else None

    @classmethod
    def _get_entity_nodes(cls, g: nx.Graph) -> List[int]:
        return [node for node, is_ent in g.nodes(data='is_ent') if is_ent]

    def _split_into_valid_subgraphs(self, ag: nx.Graph) -> List[nx.Graph]:
        utils.get_logger().debug(f'Splitting graph of size {len(ag.nodes)} into valid subgraphs..')
        node_groups = defaultdict(set)
        for node, path in nx.multi_source_dijkstra_path(ag, self._get_entity_nodes(ag)).items():
            ent_node = path[0]
            node_groups[ent_node].add(node)
        return [ag.subgraph(nodes) for nodes in node_groups.values()]


class EdinMatcher(GreedyClusteringMatcher):
    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        mention_graph = self._get_alignment_graph(eval_mode, False)
        mention_ents = self._get_top_entities_for_mentions(eval_mode)
        clusters = []
        for mention_cluster in self._get_subgraphs(mention_graph):
            mentions = self._get_mention_nodes(mention_cluster)
            ent = None
            ent_counts = Counter([mention_ents[m_id] for m_id in mentions if m_id in mention_ents])
            if ent_counts:
                top_ent, top_ent_count = ent_counts.most_common(1)[0]
                top_ent_score = top_ent_count / len(mentions)
                if top_ent_score >= .7:  # assign entity to cluster only if it is closest entity for >= 70% of mentions
                    ent = top_ent
            clusters.append((mentions, ent))
        return self._create_alignment(clusters)

    def _get_top_entities_for_mentions(self, eval_mode: str) -> Dict[MentionId, int]:
        mention_ents = defaultdict(dict)
        for (m_id, e_id), score in self.me_ca[eval_mode].get_me_candidates(True):
            if score <= self.me_threshold:
                continue
            mention_ents[m_id][e_id] = score
        mention_ents = {m_id: max(ent_scores.items(), key=lambda x: x[1])[0] for m_id, ent_scores in mention_ents.items()}
        return mention_ents
