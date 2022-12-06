from typing import Tuple, List, Union, Dict, Optional, Set
from collections import defaultdict
from tqdm import tqdm
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

    def _get_param_dict(self) -> dict:
        return super()._get_param_dict() | {'mmt': self.mm_threshold, 'met': self.me_threshold}

    def _train_model(self, train_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        clusters_by_mid, edges = self._init_clusters_and_edges(eval_mode)
        for u, v in tqdm(edges, desc='Processing edges'):
            if isinstance(v, int):  # ME edge
                c = clusters_by_mid[u]
                if c.entity is None:
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
        clusters = self._collapse_clusters(set(clusters_by_mid.values()))
        return CandidateAlignment([(c.mentions, c.entity) for c in clusters])

    def _init_clusters_and_edges(self, eval_mode: str) -> Tuple[Dict[MentionId, Cluster], List[Tuple[MentionId, Union[MentionId, int]]]]:
        utils.get_logger().debug('Initializing base graph..')
        clusters_by_mid = {}
        # find best entity match per mention
        me_edges = defaultdict(dict)
        for (m_id, e_id), score in self.me_ca[eval_mode].get_me_candidates(True):
            if m_id not in clusters_by_mid:
                clusters_by_mid[m_id] = Cluster({m_id}, None)
            if score > self.me_threshold:
                me_edges[m_id][e_id] = score
        # collect all potential edges
        edges = [(m_id, *max(ent_scores.items(), key=lambda x: x[1])) for m_id, ent_scores in me_edges.items()]
        for (m_one, m_two), score in self.mm_ca[eval_mode].get_mm_candidates(True):
            if score > self.mm_threshold:
                edges.append((m_one, m_two, score))
        ordered_edges = [(u, v) for u, v, _ in sorted(edges, key=lambda x: x[2], reverse=True)]
        return clusters_by_mid, ordered_edges

    @classmethod
    def _collapse_clusters(cls, clusters: Set[Cluster]) -> Set[Cluster]:
        cluster_by_ent = defaultdict(set)
        for c in clusters:
            cluster_by_ent[c.entity].add(c)
        collapsed_clusters = set()
        for ent, clusters in cluster_by_ent.items():
            if ent is None:
                collapsed_clusters.update(clusters)
            else:
                collapsed_clusters.add(Cluster({m for c in clusters for m in c.mentions}, ent))
        return collapsed_clusters
