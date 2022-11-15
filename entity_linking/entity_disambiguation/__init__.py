from .data import CorpusType, get_data_corpora
from .matching.util import MatchingScenario, MatchingApproach
from .matching.io import store_candidate_alignment
from .matching import initialize_matcher
import utils
from impl.subject_entity import mention_detection


def run_evaluation(scenario: MatchingScenario, approach: MatchingApproach, corpus_type: CorpusType, sample_size: int, params: dict, save_alignment: bool, save_test_alignment: bool):
    if corpus_type == CorpusType.LIST:  # make sure subject entities are initialized
        mention_detection.detect_mentions()
    matcher = initialize_matcher(scenario, approach, params)
    train_data, eval_data, test_data = get_data_corpora(corpus_type, sample_size)

    alignments = matcher.train(train_data, eval_data, save_alignment)
    alignments |= matcher.test(test_data)
    if save_alignment or save_test_alignment:
        store_candidate_alignment(matcher.get_approach_name(), alignments)
    utils.get_logger().info('DONE.')
