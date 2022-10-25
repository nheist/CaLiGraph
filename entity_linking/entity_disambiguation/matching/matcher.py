from typing import Set, List, Optional, Dict
from abc import ABC, abstractmethod
from time import process_time
import utils
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.evaluation import PrecisionRecallF1Evaluator
from entity_linking.entity_disambiguation.matching.util import MatchingScenario, load_candidates
from impl.wikipedia.page_parser import WikiListing
from impl.caligraph.entity import ClgEntity


class Matcher(ABC):
    MODE_TRAIN, MODE_EVAL, MODE_TEST = 'train', 'eval', 'test'

    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__()
        self.scenario = scenario
        self.id = params['id']

    def get_approach_name(self) -> str:
        approach_params = [self.id] + [f'{k}={v}' for k, v in self._get_param_dict().items()]
        return '_'.join(approach_params)

    def _get_param_dict(self) -> dict:
        return {}

    def train(self, train_corpus: DataCorpus, eval_corpus: DataCorpus, eval_on_train: bool) -> Dict[str, Set[Pair]]:
        utils.get_logger().info('Training matcher..')
        self._train_model(train_corpus, eval_corpus)
        return {self.MODE_TRAIN: self._evaluate(self.MODE_TRAIN, train_corpus)} if eval_on_train else {}

    @abstractmethod
    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass

    def test(self, test_corpus: DataCorpus) -> Dict[str, Set[Pair]]:
        utils.get_logger().info('Testing matcher..')
        return {self.MODE_TEST: self._evaluate(self.MODE_TEST, test_corpus)}

    def _evaluate(self, eval_mode: str, data_corpus: DataCorpus) -> Dict[MatchingScenario, Set[Pair]]:
        utils.release_gpu()
        predictions = {}
        source = data_corpus.get_listings()
        if self.scenario.is_MM():
            scenario = MatchingScenario.MENTION_MENTION
            predictions[scenario] = self._evaluate_scenario(eval_mode, scenario, source, None, data_corpus.mm_alignment)
        if self.scenario.is_ME():
            scenario = MatchingScenario.MENTION_ENTITY
            target = data_corpus.get_entities()
            predictions[scenario] = self._evaluate_scenario(eval_mode, scenario, source, target, data_corpus.me_alignment)
        return predictions

    def _evaluate_scenario(self, eval_mode: str, scenario: MatchingScenario, source, target, alignment) -> Set[Pair]:
        pred_start = process_time()
        prediction = self.predict(eval_mode, source, target)
        prediction_time_in_seconds = int(process_time() - pred_start)

        evaluator = PrecisionRecallF1Evaluator(self.get_approach_name(), scenario)
        evaluator.compute_and_log_metrics(eval_mode, prediction, alignment, prediction_time_in_seconds)
        return prediction

    @abstractmethod
    def predict(self, eval_mode: str, source: List[WikiListing], target: Optional[List[ClgEntity]]) -> Set[Pair]:
        pass


class MatcherWithCandidates(Matcher, ABC):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        if params['mm_approach']:
            self.mm_approach = params['mm_approach']
            self.mm_candidates = load_candidates(self.mm_approach, MatchingScenario.MENTION_MENTION)
        if params['me_approach']:
            self.me_approach = params['me_approach']
            self.me_candidates = load_candidates(self.me_approach, MatchingScenario.MENTION_ENTITY)

    def _get_param_dict(self) -> dict:
        params = {}
        if hasattr(self, 'mm_approach'):
            params['mma'] = self.mm_approach
        if hasattr(self, 'me_approach'):
            params['mea'] = self.me_approach
        return super()._get_param_dict() | params
