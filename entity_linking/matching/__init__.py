from entity_linking.matching.matcher import Matcher
from entity_linking.matching.util import MatchingScenario, MatchingApproach
from entity_linking.matching.lexical import ExactMatcher, WordMatcher, BM25Matcher
from entity_linking.matching.graph import PopularityMatcher
from entity_linking.matching.biencoder import BiEncoderMatcher
from entity_linking.matching.crossencoder import CrossEncoderMatcher
from entity_linking.matching.tdfusion import TopDownFusionMatcher
from entity_linking.matching.bufusion import BottomUpFusionMatcher, NastyLinker
import utils


def initialize_matcher(scenario: MatchingScenario, approach: MatchingApproach, params: dict) -> Matcher:
    utils.get_logger().info('Initializing matcher..')
    if approach == MatchingApproach.EXACT:
        matcher_factory = ExactMatcher
    elif approach == MatchingApproach.WORD:
        matcher_factory = WordMatcher
    elif approach == MatchingApproach.BM25:
        matcher_factory = BM25Matcher
    elif approach == MatchingApproach.POPULARITY:
        matcher_factory = PopularityMatcher
    elif approach == MatchingApproach.BIENCODER:
        matcher_factory = BiEncoderMatcher
    elif approach == MatchingApproach.CROSSENCODER:
        matcher_factory = CrossEncoderMatcher
    elif approach == MatchingApproach.TOP_DOWN_FUSION:
        matcher_factory = TopDownFusionMatcher
    elif approach == MatchingApproach.BOTTOM_UP_FUSION:
        matcher_factory = BottomUpFusionMatcher
    elif approach == MatchingApproach.NASTY_LINKER:
        matcher_factory = NastyLinker
    else:
        raise ValueError(f'Matching approach not implemented: {approach.value}')
    return matcher_factory(scenario, params)