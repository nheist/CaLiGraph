from entity_linking.entity_disambiguation.matching.matcher import Matcher
from entity_linking.entity_disambiguation.matching.util import MatchingScenario, MatchingApproach, load_candidates
from entity_linking.entity_disambiguation.matching.lexical import ExactMatcher, WordMatcher
from entity_linking.entity_disambiguation.matching.graph import PopularityMatcher
from entity_linking.entity_disambiguation.matching.biencoder import BiEncoderMatcher
from entity_linking.entity_disambiguation.matching.crossencoder import CrossEncoderMatcher
from entity_linking.entity_disambiguation.matching.tdfusion import WeakestMentionMatcher, WeakestEntityMatcher, WeakestLinkMatcher, PrecisionWeightedWeakestLinkMatcher
from entity_linking.entity_disambiguation.matching.bufusion import BottomUpFusionMatcher
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
    elif approach == MatchingApproach.WEAKEST_MENTION:
        matcher_factory = WeakestMentionMatcher
    elif approach == MatchingApproach.WEAKEST_ENTITY:
        matcher_factory = WeakestEntityMatcher
    elif approach == MatchingApproach.WEAKEST_LINK:
        matcher_factory = WeakestLinkMatcher
    elif approach == MatchingApproach.PRECISION_WEIGHTED_WEAKEST_LINK:
        matcher_factory = PrecisionWeightedWeakestLinkMatcher
    elif approach == MatchingApproach.BOTTOM_UP_FUSION:
        matcher_factory = BottomUpFusionMatcher
    else:
        raise ValueError(f'Matching approach not implemented: {approach.value}')
    return matcher_factory(scenario, params)
