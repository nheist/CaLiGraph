import utils
from . import data
from .matching import util as matching_util
from .matching import MatchingScenario, MatchingApproach, initialize_matcher


def run_evaluation(scenario: MatchingScenario, approach: MatchingApproach, params: dict, save_alignment: bool):
    train_corpus, eval_corpus, test_corpus = _get_train_val_test_corpora_for_scenario(scenario)
    matcher = initialize_matcher(scenario, approach, params)
    if save_alignment:
        train_alignment = matcher.train(train_corpus)
        eval_alignment = matcher.eval(eval_corpus)
        test_alignment = matcher.test(test_corpus)
        matching_util.store_candidates(matcher.get_approach_id(), {
            matcher.MODE_TRAIN: train_alignment,
            matcher.MODE_EVAL: eval_alignment,
            matcher.MODE_TEST: test_alignment
        })
    else:
        matcher.train(train_corpus)
        matcher.eval(eval_corpus)
        matcher.test(test_corpus)


def _get_train_val_test_corpora_for_scenario(scenario: MatchingScenario):
    match scenario:
        case MatchingScenario.MENTION_MENTION:
            return utils.load_or_create_cache('MM_datasets', data.get_mm_train_val_test_corpora)
        case MatchingScenario.MENTION_ENTITY:
            return utils.load_or_create_cache('ME_datasets', data.get_me_train_val_test_corpora)
