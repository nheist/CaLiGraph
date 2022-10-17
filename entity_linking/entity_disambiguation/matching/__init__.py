from entity_linking.entity_disambiguation.matching.matcher import Matcher
from entity_linking.entity_disambiguation.matching.util import MatchingScenario, MatchingApproach, load_candidates
from entity_linking.entity_disambiguation.matching.lexical import ExactMatcher, WordMatcher
from entity_linking.entity_disambiguation.matching.graph import PopularityMatcher
from entity_linking.entity_disambiguation.matching.biencoder import BiEncoderMatcher
from entity_linking.entity_disambiguation.matching.crossencoder import CrossEncoderMatcher
import utils


def initialize_matcher(scenario: MatchingScenario, approach: MatchingApproach, params: dict) -> Matcher:
    utils.get_logger().info('Initializing matcher..')
    if approach == MatchingApproach.EXACT:
        matcher_factory = ExactMatcher
    elif approach == MatchingApproach.WORD:
        matcher_factory = WordMatcher
    elif approach == MatchingApproach.POPULARITY:
        matcher_factory = PopularityMatcher
    elif approach == MatchingApproach.BIENCODER:
        matcher_factory = BiEncoderMatcher
    elif approach == MatchingApproach.CROSSENCODER:
        matcher_factory = CrossEncoderMatcher
    else:
        raise ValueError(f'Matching approach not implemented: {approach.value}')
    return matcher_factory(scenario, params)
