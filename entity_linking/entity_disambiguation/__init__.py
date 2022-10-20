import utils
from . import data
from .matching.util import MatchingScenario, MatchingApproach, store_candidates
from .matching import initialize_matcher
from impl.subject_entity import mention_detection


def run_evaluation(scenario: MatchingScenario, approach: MatchingApproach, params: dict, save_alignment: bool):
    mention_detection.detect_mentions()  # make sure subject entities are initialized
    matcher = initialize_matcher(scenario, approach, params)
    if scenario == MatchingScenario.FUSION:
        mm_test = _get_train_val_test_corpora_for_scenario(MatchingScenario.MENTION_MENTION)[2]
        me_test = _get_train_val_test_corpora_for_scenario(MatchingScenario.MENTION_ENTITY)[2]
        matcher.test(mm_test, me_test)
    else:
        train_corpus, eval_corpus, test_corpus = _get_train_val_test_corpora_for_scenario(scenario)
        if save_alignment:
            alignments = matcher.train(train_corpus, eval_corpus, save_alignment)
            alignments |= matcher.test(test_corpus)
            utils.get_logger().info('Storing candidates..')
            store_candidates(matcher.get_approach_name(), alignments)
        else:
            matcher.train(train_corpus, eval_corpus, save_alignment)
            matcher.test(test_corpus)


def _get_train_val_test_corpora_for_scenario(scenario: MatchingScenario):
    utils.get_logger().info('Loading training data..')
    if scenario == MatchingScenario.MENTION_MENTION:
        return utils.load_or_create_cache('MM_datasets', data.get_mm_train_val_test_corpora)
    elif scenario == MatchingScenario.MENTION_ENTITY:
        return utils.load_or_create_cache('ME_datasets', data.get_me_train_val_test_corpora)
    raise ValueError(f'Invalid scenario: {scenario.name}')
