from typing import Set, Tuple
from collections import defaultdict
from entity_linking.entity_disambiguation.data import CandidateAlignment, DataCorpus
from entity_linking.entity_disambiguation.matching.matcher import MatcherWithCandidates
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from impl.caligraph.entity import ClgEntity


class PopularityMatcher(MatcherWithCandidates):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        self.entity_popularity = None

    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        self.entity_popularity = self._compute_entity_popularity(train_corpus.get_entities())

    @classmethod
    def _compute_entity_popularity(cls, entities: Set[ClgEntity]) -> dict:
        assert entities is not None, 'PopularityMatcher can only be applied to corpus with entities.'
        entity_popularity = defaultdict(int)
        for ent in entities:
            out_degree = len(ent.get_properties(as_tuple=True))
            in_degree = len(ent.get_inverse_properties(as_tuple=True))
            entity_popularity[ent.idx] = out_degree + in_degree
        return entity_popularity

    def _get_entity_popularity(self, ent_with_score: Tuple[int, float]) -> int:
        ent_idx = ent_with_score[0]
        return self.entity_popularity[ent_idx]

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        assert self.scenario == MatchingScenario.MENTION_ENTITY, 'PopularityMatcher can only be applied in ME scenario.'
        candidates_by_mention = defaultdict(set)
        for (mention_id, ent_id), score in self.me_ca[eval_mode].get_me_candidates(True):
            candidates_by_mention[mention_id].add((ent_id, score))
        ca = CandidateAlignment()
        for mention_id, ents in candidates_by_mention.items():
            ent_id, score = max(ents, key=self._get_entity_popularity)
            ca.add_candidate((mention_id, ent_id), score)
        return ca
