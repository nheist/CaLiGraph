import utils
from . import data
from .matching.util import MatchingScenario, MatchingApproach, store_candidates
from .matching import initialize_matcher
from impl.subject_entity import mention_detection


def run_evaluation(scenario: MatchingScenario, approach: MatchingApproach, params: dict, save_alignment: bool):
    mention_detection.detect_mentions()  # make sure subject entities are initialized
    matcher = initialize_matcher(scenario, approach, params)
    train_data, eval_data, test_data = utils.load_or_create_cache('ED_datasets', data.get_train_val_test_corpora)

    alignments = matcher.train(train_data, eval_data, save_alignment)
    alignments |= matcher.test(test_data)
    if save_alignment:
        store_candidates(matcher.get_approach_name(), alignments)
