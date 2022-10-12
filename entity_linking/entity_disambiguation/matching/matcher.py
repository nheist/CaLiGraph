from typing import Set, List, Optional, Dict
from abc import ABC, abstractmethod
from time import process_time
import utils
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.evaluation import PrecisionRecallF1Evaluator
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from impl.wikipedia.page_parser import WikiListing
from impl.caligraph.entity import ClgEntity


class Matcher(ABC):
    MODE_TRAIN, MODE_EVAL, MODE_TEST = 'train', 'eval', 'test'

    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__()
        self.scenario = scenario
        self.version = params['version']

    def get_approach_id(self) -> str:
        approach_params = [self.scenario.value, f'v{self.version}', type(self).__name__]
        approach_params += [f'{k}={v}' for k, v in self._get_param_dict().items()]
        return '_'.join(approach_params)

    def _get_param_dict(self) -> dict:
        return {}

    def train(self, training_set: DataCorpus, eval_set: DataCorpus, save_alignment: bool) -> Dict[str, Set[Pair]]:
        utils.get_logger().info('Training matcher..')
        self._train_model(training_set, eval_set)
        return {self.MODE_TRAIN: self._evaluate(self.MODE_TRAIN, training_set)} if save_alignment else {}

    @abstractmethod
    def _train_model(self, training_set: DataCorpus, eval_set: DataCorpus):
        pass

    def test(self, test_set: DataCorpus) -> Dict[str, Set[Pair]]:
        utils.get_logger().info('Testing matcher..')
        return {self.MODE_TEST: self._evaluate(self.MODE_TEST, test_set)}

    def _evaluate(self, prefix: str, data_corpus: DataCorpus) -> Set[Pair]:
        utils.release_gpu()
        pred_start = process_time()
        prediction = self.predict(prefix, data_corpus.source, data_corpus.target)
        prediction_time_in_seconds = int(process_time() - pred_start)

        evaluator = PrecisionRecallF1Evaluator(self.get_approach_id(), self.scenario)
        evaluator.compute_and_log_metrics(prefix, prediction, data_corpus.alignment, prediction_time_in_seconds)
        return prediction

    @abstractmethod
    def predict(self, prefix: str, source: List[WikiListing], target: Optional[List[ClgEntity]]) -> Set[Pair]:
        pass


class MatcherWithCandidates(Matcher, ABC):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        self.blocking_approach = params['blocking_approach']
        self.candidates = params['candidates']

    def _get_param_dict(self) -> dict:
        return super()._get_param_dict() | {'b': self.blocking_approach}
