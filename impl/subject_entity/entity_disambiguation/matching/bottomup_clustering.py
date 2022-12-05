from typing import Tuple, List, Union, Dict
from collections import defaultdict, namedtuple
from tqdm import tqdm
import utils
from impl.wikipedia import MentionId
from impl.subject_entity.entity_disambiguation.data import CandidateAlignment, DataCorpus
from impl.subject_entity.entity_disambiguation.matching.util import MatchingScenario
from impl.subject_entity.entity_disambiguation.matching.matcher import MatcherWithCandidates


Cluster = namedtuple('Cluster', ['mentions', 'entity'])


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
                c_common = Cluster(c_one.mentions | c_two.mentions, c_one.entity if c_two.entity is None else c_two.entity)
                for m_id in c_common.mentions:
                    clusters_by_mid[m_id] = c_common
        clusters = [(c.mentions, c.entity) for c in set(clusters_by_mid.values())]
        return CandidateAlignment(clusters)

    def _init_clusters_and_edges(self, eval_mode: str) -> Tuple[Dict[MentionId, Cluster], List[Tuple[MentionId, Union[MentionId, int]]]]:
        utils.get_logger().debug('Initializing base graph..')
        # find best entity match per mention
        me_edges = defaultdict(dict)
        for (m_id, e_id), score in self.me_ca[eval_mode].get_me_candidates(True):
            if score > self.me_threshold:
                me_edges[m_id][e_id] = score
        # init clusters
        clusters_by_mid = {m_id: Cluster({m_id}, None) for m_id in me_edges}
        # collect all potential edges
        edges = [(m_id, *max(ent_scores.items(), key=lambda x: x[1])) for m_id, ent_scores in me_edges.items()]
        for (m_one, m_two), score in self.mm_ca[eval_mode].get_mm_candidates(True):
            if score > self.mm_threshold:
                edges.append((m_one, m_two, score))
        ordered_edges = [(u, v) for u, v, _ in sorted(edges, key=lambda x: x[2], reverse=True)]
        return clusters_by_mid, ordered_edges
