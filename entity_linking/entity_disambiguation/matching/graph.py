from typing import List, Optional, Set
from collections import defaultdict
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.matcher import MatcherWithCandidates, MatchingScenario
from impl.caligraph.entity import ClgEntity
from impl.wikipedia.page_parser import WikiListing


class PopularityMatcher(MatcherWithCandidates):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        self.entity_popularity = None

    def _train_model(self, training_set: DataCorpus):
        self.entity_popularity = self._compute_entity_popularity(training_set.target)

    @classmethod
    def _compute_entity_popularity(cls, entities: List[ClgEntity]) -> dict:
        assert entities is not None, 'PopularityMatcher can only be applied to corpus with entities.'
        entity_popularity = defaultdict(int)
        for ent in entities:
            out_degree = len(ent.get_properties(as_tuple=True))
            in_degree = len(ent.get_inverse_properties(as_tuple=True))
            entity_popularity[ent.idx] = out_degree + in_degree
        return entity_popularity

    def _get_entity_popularity(self, ent_idx: int) -> int:
        return self.entity_popularity[ent_idx]

    def predict(self, prefix: str, source: List[WikiListing], target: Optional[List[ClgEntity]]) -> Set[Pair]:
        assert target is not None, 'PopularityMatcher can only be applied to corpus with target.'
        candidates_by_source = defaultdict(set)
        for item, ent in self.candidates[prefix]:
            candidates_by_source[item].add(ent)
        alignment = {item: max(ents, key=self._get_entity_popularity) for item, ents in candidates_by_source.items()}
        return {Pair(source_item, target_entity) for source_item, target_entity in alignment.items()}
