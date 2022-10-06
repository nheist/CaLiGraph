from entity_linking.entity_disambiguation.matching import util as matching_util
from entity_linking.entity_disambiguation.matching.matcher import MatchingScenario, MatchingApproach, Matcher
from entity_linking.entity_disambiguation.matching.lexical import ExactMatcher, WordMatcher
from entity_linking.entity_disambiguation.matching.graph import PopularityMatcher
from entity_linking.entity_disambiguation.matching.biencoder import BiEncoderMatcher


def initialize_matcher(scenario: MatchingScenario, approach: MatchingApproach, params: dict) -> Matcher:
    # prepare candidates (if necessary)
    if params['blocking_approach'] is not None:
        params['candidates'] = matching_util.load_candidates(params['blocking_approach'])
    # initialize main matcher
    if approach == MatchingApproach.EXACT:
        matcher_factory = ExactMatcher
    elif approach == MatchingApproach.WORD:
        matcher_factory = WordMatcher
    elif approach == MatchingApproach.POPULARITY:
        matcher_factory = PopularityMatcher
    elif approach == MatchingApproach.BIENCODER:
        matcher_factory = BiEncoderMatcher
    else:
        raise ValueError(f'Blocking approach not implemented: {approach.value}')
    return matcher_factory(scenario, params)
