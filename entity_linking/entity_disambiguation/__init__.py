from .data import CorpusType, get_data_corpora
from .matching.util import MatchingScenario, MatchingApproach, store_candidates
from .matching import initialize_matcher
from impl.subject_entity import mention_detection


def run_evaluation(scenario: MatchingScenario, approach: MatchingApproach, corpus_type: CorpusType, params: dict, save_alignment: bool):
    mention_detection.detect_mentions()  # make sure subject entities are initialized
    matcher = initialize_matcher(scenario, approach, params)
    train_data, eval_data, test_data = get_data_corpora(corpus_type)

    alignments = matcher.train(train_data, eval_data, save_alignment)
    alignments |= matcher.test(test_data)
    if save_alignment:
        store_candidates(matcher.get_approach_name(), alignments)
