import utils
from . import data
from .matching import util as matching_util
from .matching import MatchingScenario, MatchingApproach, initialize_matcher


def run_evaluation(scenario: MatchingScenario, approach: MatchingApproach, params: dict, save_alignment: bool):
    train_corpus, eval_corpus, test_corpus = _get_train_val_test_corpora_for_scenario(scenario)
    matcher = initialize_matcher(scenario, approach, params)
    if save_alignment:
        alignments = matcher.train(train_corpus, eval_corpus)
        alignments |= matcher.test(test_corpus)
        matching_util.store_candidates(matcher.get_approach_id(), alignments)
    else:
        matcher.train(train_corpus, eval_corpus)
        matcher.test(test_corpus)


def _get_train_val_test_corpora_for_scenario(scenario: MatchingScenario):
    print(scenario)
    match scenario:
        case MatchingScenario.MENTION_MENTION:
            return utils.load_or_create_cache('MM_datasets', data.get_mm_train_val_test_corpora)
        case MatchingScenario.MENTION_ENTITY:
            return utils.load_or_create_cache('ME_datasets', data.get_me_train_val_test_corpora)
        case _:
            raise ValueError(f'Invalid scenario: {scenario.name}')
