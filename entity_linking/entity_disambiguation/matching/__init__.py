from entity_linking.entity_disambiguation.matching import util as matching_util
from entity_linking.entity_disambiguation.matching.matcher import MatchingScenario, MatchingApproach, Matcher
from entity_linking.entity_disambiguation.matching.lexical import ExactMatcher, WordMatcher
from entity_linking.entity_disambiguation.matching.graph import PopularityMatcher


def initialize_matcher(scenario: MatchingScenario, approach: MatchingApproach, params: dict) -> Matcher:
    # prepare candidates (if necessary)
    if 'blocking_approach' in params:
        params['candidates'] = matching_util.load_candidates(params['blocking_approach'])
    # initialize main matcher
    match approach:
        case MatchingApproach.EXACT:
            matcher_factory = ExactMatcher
        case MatchingApproach.WORD:
            matcher_factory = WordMatcher
        case MatchingApproach.POPULARITY:
            matcher_factory = PopularityMatcher
        case _:
            raise ValueError(f'Blocking approach not implemented: {approach.value}')
    return matcher_factory(scenario, params)
