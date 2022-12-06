from typing import List, Optional, Set, Iterable
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import utils
from impl.wikipedia import MentionId
from impl.subject_entity.entity_disambiguation.data import CandidateAlignment, DataCorpus
from impl.subject_entity.entity_disambiguation.matching.util import MatchingScenario
from impl.subject_entity.entity_disambiguation.matching.matcher import MatcherWithCandidates


class Cluster:
    mentions: set
    entity: Optional[int]

    def __init__(self, mentions: set, entity: Optional[int]):
        self.mentions = mentions
        self.entity = entity


class BottomUpClusteringMatcher(MatcherWithCandidates):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.mm_threshold = params['mm_threshold']
        self.me_threshold = params['me_threshold']
        self.me_cluster_threshold = params['me_cluster_threshold']

    def _get_param_dict(self) -> dict:
        return super()._get_param_dict() | {'mmt': self.mm_threshold, 'met': self.me_threshold, 'mct': self.me_cluster_threshold}

    def _train_model(self, train_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        # initialize clusters and edges
        clusters_by_mid = {}
        me_edges = defaultdict(dict)  # find best entity match per mention
        for (m_id, e_id), score in self.me_ca[eval_mode].get_me_candidates(True):
            if m_id not in clusters_by_mid:
                clusters_by_mid[m_id] = Cluster({m_id}, None)
            if score > self.me_threshold:
                me_edges[m_id][e_id] = score
        # collect all valid edges and sort by score
        me_edges = {m_id: max(ent_scores.items(), key=lambda x: x[1]) for m_id, ent_scores in me_edges.items()}
        edges = [(m_id, e_id, score) for m_id, (e_id, score) in me_edges.items()]
        for (m_one, m_two), score in self.mm_ca[eval_mode].get_mm_candidates(True):
            if score > self.mm_threshold:
                edges.append((m_one, m_two, score))
        edges = sorted(edges, key=lambda x: x[2], reverse=True)
        # process edges
        for u, v, score in tqdm(edges, desc='Processing edges'):
            if isinstance(v, int):  # ME edge
                c = clusters_by_mid[u]
                if c.entity is not None:
                    continue
                c.entity = v
            else:  # MM edge
                c_one, c_two = clusters_by_mid[u], clusters_by_mid[v]
                if c_one.entity is not None and c_two.entity is not None:
                    continue
                if len(c_one.mentions) < len(c_two.mentions):
                    c_one, c_two = c_two, c_one  # merge smaller cluster into bigger one
                c_one.mentions = c_one.mentions | c_two.mentions
                if c_one.entity is None:
                    c_one.entity = c_two.entity
                for m_id in c_two.mentions:
                    clusters_by_mid[m_id] = c_one
        clusters = set(clusters_by_mid.values())
        if self.me_cluster_threshold > 0:
            self._filter_clusters_by_entity_frequency(clusters, me_edges)
        clusters = self._collapse_clusters(clusters)
        # initialize and return candidate alignment
        return CandidateAlignment([(c.mentions, c.entity) for c in clusters])

    @classmethod
    def _collapse_clusters(cls, clusters: Set[Cluster]) -> Set[Cluster]:
        cluster_by_ent = defaultdict(set)
        for c in clusters:
            cluster_by_ent[c.entity].add(c)
        collapsed_clusters = set()
        for ent, clusters in cluster_by_ent.items():
            if ent is None:
                collapsed_clusters.update(clusters)
            collapsed_clusters.add(Cluster({m for c in clusters for m in c.mentions}, ent))
        return collapsed_clusters

    def _filter_clusters_by_entity_frequency(self, clusters: Set[Cluster], me_edges: dict):
        for c in clusters:
            ent_count = sum(1 for m in c.mentions if m in me_edges and c.entity == me_edges[m][0])
            ent_freq = ent_count / len(c.mentions)
            if ent_freq <= self.me_cluster_threshold:
                c.entity = None

    def _get_alignment_graph(self, eval_mode: str) -> nx.Graph:
        utils.get_logger().debug('Initializing alignment graph..')
        ag = nx.Graph()
        for (m_id, e_id), score in self.me_ca[eval_mode].get_me_candidates(True):
            ag.add_node(m_id, is_ent=False)
            if score > self.me_threshold:
                ag.add_node(e_id, is_ent=True)
                ag.add_edge(m_id, e_id, weight=min(score, 1))
        ag.add_weighted_edges_from([(u, v, min(score, 1)) for (u, v), score in self.mm_ca[eval_mode].get_mm_candidates(True) if score > self.mm_threshold])
        return ag

    @classmethod
    def _get_entity_node(cls, g: nx.Graph) -> Optional[int]:
        ent_nodes = cls._get_entity_nodes(g)
        return ent_nodes[0] if ent_nodes else None

    @classmethod
    def _get_entity_nodes(cls, g: nx.Graph) -> List[int]:
        return [node for node, is_ent in g.nodes(data='is_ent') if is_ent]

    @classmethod
    def _get_mention_nodes(cls, g: nx.Graph) -> Set[MentionId]:
        return {node for node, is_ent in g.nodes(data='is_ent') if not is_ent}

    def _get_subgraphs(self, ag: nx.Graph) -> Iterable[nx.Graph]:
        for nodes in nx.connected_components(ag):
            yield ag.subgraph(nodes)
