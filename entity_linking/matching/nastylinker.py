from typing import Set, Dict, Optional, Tuple, List
from collections import defaultdict
import random
import itertools
import numpy as np
import utils
from impl.wikipedia.page_parser import MentionId
from entity_linking.data import CandidateAlignment, DataCorpus, Pair
from entity_linking.matching.util import MatchingScenario
from entity_linking.matching.crossencoder import CrossEncoderMatcher


class MentionCluster:
    idx: int
    mentions: Set[MentionId]
    candidates: Dict[MentionId, float]
    entity: Optional[int]

    def __init__(self, idx: int, mentions: Set[MentionId], candidates: Dict[MentionId, float], entity: Optional[int] = None):
        self.idx = idx
        self.mentions = mentions
        self.candidates = candidates
        self.entity = entity


class NastyLinker(CrossEncoderMatcher):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        self.cluster_comparisons = params['cluster_comparisons']

    def _get_param_dict(self) -> dict:
        return super()._get_param_dict() | {'cc': self.cluster_comparisons}

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        # prepare inputs
        mention_input, _ = data_corpus.get_mention_input(self.add_page_context, self.add_text_context)
        entity_input = data_corpus.get_entity_input(self.add_entity_abstract, self.add_kg_info)
        # retrieve entities for mentions and form clusters based on best entities; init remaining mentions as 1-clusters
        me_predictions = self._make_me_predictions(eval_mode, mention_input, entity_input)
        clusters, cluster_by_mid = self._init_clusters(eval_mode, me_predictions)
        # find cluster merges from mention-mention candidates that are transformable to mention-entity matches
        utils.get_logger().debug('Merging mention clusters with entity clusters..')
        iteration = 1
        while True:  # repeat as long as we have mc/ec-candidates
            me_candidates, me_clusters = self._find_me_cluster_merges(clusters, cluster_by_mid, me_predictions)
            if not me_candidates:
                break
            me_scores = self._compute_scores_for_candidates(me_candidates, mention_input, entity_input)
            me_cluster_scores = dict(zip(me_clusters, me_scores))
            self._merge_clusters(me_cluster_scores, self.me_threshold, clusters, cluster_by_mid, iteration)
            iteration += 1
        # merge remaining mention clusters
        utils.get_logger().debug('Merging mention clusters with mention clusters..')
        iteration = 1
        while True:  # repeat as long as we have mc/mc-candidates
            mm_candidates, mm_clusters = self._find_mm_cluster_merges(clusters, cluster_by_mid)
            if not mm_candidates:
                break
            mm_scores = self._compute_scores_for_candidates(mm_candidates, mention_input, entity_input)
            mm_cluster_scores = self._compute_cluster_merge_scores(mm_scores, mm_clusters)
            self._merge_clusters(mm_cluster_scores, self.mm_threshold, clusters, cluster_by_mid, iteration)
            iteration += 1
        # compute final alignment
        ca = CandidateAlignment()
        for cluster in clusters:
            for m_id in cluster.mentions:
                ca.add_candidate((m_id, cluster.entity), 1)
            for pair in itertools.combinations(cluster.mentions, 2):
                ca.add_candidate(pair, 1)
        ca.add_entity_clustering([cluster.mentions for cluster in clusters])
        return ca

    def _make_me_predictions(self, eval_mode: str, mention_input: Dict[MentionId, str], entity_input: Dict[int, str]) -> Dict[MentionId, Dict[int, float]]:
        utils.get_logger().debug('Computing mention-entity matches..')
        candidates = list(self.me_ca[eval_mode].get_me_candidates(False))
        candidate_scores = self._compute_scores_for_candidates(candidates, mention_input, entity_input)
        predictions = defaultdict(dict)
        for (mention_id, entity_id), score in zip(candidates, candidate_scores):
            predictions[mention_id][entity_id] = score
        return predictions

    def _compute_scores_for_candidates(self, candidates: List[Pair], mention_input: Dict[MentionId, str], entity_input: Dict[int, str]) -> List[float]:
        model_input = []
        for pair in candidates:
            if isinstance(pair[1], MentionId):
                mention_a, mention_b = pair
                model_input.append([mention_input[mention_a], mention_input[mention_b]])
            else:
                mention_id, entity_id = pair
                model_input.append([mention_input[mention_id], entity_input[entity_id]])
        utils.release_gpu()
        return self.model.predict(model_input, batch_size=self.batch_size, show_progress_bar=True)

    def _init_clusters(self, eval_mode: str, predictions: Dict[MentionId, Dict[int, float]]) -> Tuple[Set[MentionCluster], Dict[MentionId, MentionCluster]]:
        utils.get_logger().debug('Initializing clusters..')
        # group mentions by matching entity
        me_mapping = {m_id: max(ents.items(), key=lambda x: x[1]) for m_id, ents in predictions.items()}
        me_mapping = {m_id: ent_id for m_id, (ent_id, score) in me_mapping.items() if score > self.me_threshold}
        em_mapping = defaultdict(set)
        for m_id, e_id in me_mapping.items():
            em_mapping[e_id].add(m_id)
        # group candidates by mention
        mention_candidates = defaultdict(dict)
        for (mention_a, mention_b), score in self.mm_ca[eval_mode].get_mm_candidates(True):
            mention_candidates[mention_a][mention_b] = score
            mention_candidates[mention_b][mention_a] = score
        # form clusters with known entities
        clusters = set()
        cluster_by_mid = {}
        cluster_id = 0
        for e_id, mention_ids in em_mapping.items():
            cluster = MentionCluster(cluster_id, mention_ids, defaultdict(float), e_id)
            clusters.add(cluster)
            for m_id in mention_ids:
                cluster_by_mid[m_id] = cluster
            cluster_id += 1
        # form clusters with remaining mentions
        for m_id, candidates in mention_candidates.items():
            if m_id in cluster_by_mid:
                continue
            cluster = MentionCluster(cluster_id, {m_id}, defaultdict(float, candidates))
            clusters.add(cluster)
            cluster_by_mid[m_id] = cluster
            cluster_id += 1
        return clusters, cluster_by_mid

    def _find_me_cluster_merges(self, clusters: Set[MentionCluster], cluster_by_mid: Dict[MentionId, MentionCluster], me_predictions: Dict[MentionId, Dict[int, float]]) -> Tuple[List[Pair], List[Tuple[int, int]]]:
        me_candidates, me_clusters = [], []
        for cluster in clusters:
            if cluster.entity is not None:
                continue  # only use mention clusters to find merge candidates
            ent_cluster_candidates = {cluster_by_mid[m_id]: score for m_id, score in cluster.candidates.items() if cluster_by_mid[m_id].entity is not None}
            if not ent_cluster_candidates:
                continue  # discard, if no entity clusters in candidate list
            top_ent_cluster = max(ent_cluster_candidates.items(), key=lambda x: x[1])[0]
            e_id = top_ent_cluster.entity
            m_id = random.choice(list(cluster.mentions))  # cluster should only have one mention, but just to make sure
            if m_id in me_predictions and e_id in me_predictions[m_id]:
                continue  # discard, as mention would already be in cluster of entity, if score was high enough
            me_candidates.append((m_id, e_id))
            me_clusters.append(tuple(sorted([cluster.idx, top_ent_cluster.idx])))
        return me_candidates, me_clusters

    def _find_mm_cluster_merges(self, clusters: Set[MentionCluster], cluster_by_mid: Dict[MentionId, MentionCluster]) -> Tuple[List[Pair], List[Tuple[int, int]]]:
        # use mention clusters to find top mention-cluster to merge with
        # make sure that cluster merge ids are sorted
        mm_candidates, mm_clusters = [], []
        for cluster in clusters:
            if cluster.entity is not None:
                continue  # only use mention clusters to find merge candidates
            if not cluster.candidates:
                continue  # discard clusters with empty candidates
            candidate_cluster = cluster_by_mid[max(cluster.candidates.items(), key=lambda x: x[1])[0]]
            # sample mention matches from clusters
            cluster_mentions = random.sample(list(cluster.mentions), min(len(cluster.mentions), self.cluster_comparisons))
            candidate_cluster_mentions = random.sample(list(candidate_cluster.mentions), min(len(candidate_cluster.mentions), self.cluster_comparisons))
            mention_candidates = list(itertools.product(cluster_mentions, candidate_cluster_mentions))
            mm_candidates.extend(mention_candidates)
            cluster_ids = tuple(sorted([cluster.idx, candidate_cluster.idx]))
            mm_clusters.extend([cluster_ids] * len(mention_candidates))
        return mm_candidates, mm_clusters

    def _compute_cluster_merge_scores(self, candidate_scores: List[float], candidate_clusters: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        cluster_merge_scores = defaultdict(list)
        for cluster_merge_id, score in zip(candidate_clusters, candidate_scores):
            cluster_merge_scores[cluster_merge_id].append(score)
        return {cluster_merge_id: np.mean(scores) for cluster_merge_id, scores in cluster_merge_scores.items()}

    def _merge_clusters(self, cluster_merge_scores: Dict[Tuple[int, int], float], threshold: float, clusters: Set[MentionCluster], cluster_by_mid: Dict[MentionId, MentionCluster], iteration: int = 0):
        cluster_by_id = {cluster.idx: cluster for cluster in clusters}
        merge_conducted, merge_discarded = 0, 0
        # merge clusters starting with highest left cluster index (in the case of a merge, we keep the left index)
        for (cluster_a_id, cluster_b_id), score in sorted(cluster_merge_scores.items(), key=lambda x: x[0][0], reverse=True):
            cluster_a, cluster_b = cluster_by_id[cluster_a_id], cluster_by_id[cluster_b_id]
            if cluster_a == cluster_b:
                continue
            if score > threshold:  # merge clusters and update indices
                cluster_a.mentions |= cluster_b.mentions
                merged_candidates = (set(cluster_a.candidates) | set(cluster_b.candidates)).difference(cluster_a.mentions)
                cluster_a.candidates = defaultdict(float, {cand: max(cluster_a.candidates[cand], cluster_b.candidates[cand]) for cand in merged_candidates})
                clusters.discard(cluster_b)
                cluster_by_id[cluster_b_id] = cluster_a
                for m_id in cluster_b.mentions:
                    cluster_by_mid[m_id] = cluster_a
                merge_conducted += 1
            else:  # make sure clusters are not considered for merge again (by deleting candidates in other cluster)
                cluster_a.candidates = defaultdict(float, {cand: score for cand, score in cluster_a.candidates.items() if cand not in cluster_b.mentions})
                cluster_b.candidates = defaultdict(float, {cand: score for cand, score in cluster_b.candidates.items() if cand not in cluster_a.mentions})
                merge_discarded += 1
        utils.get_logger().debug(f'MERGE {iteration} | Clusters: {len(clusters)}; Candidates: {len(cluster_merge_scores)}; Merged: {merge_conducted}; Discarded: {merge_discarded}')
