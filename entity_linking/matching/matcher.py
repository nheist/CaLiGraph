from typing import Dict
from abc import ABC, abstractmethod
from datetime import datetime
import utils
from entity_linking.data import CandidateAlignment, DataCorpus
from entity_linking.evaluation import PrecisionRecallF1Evaluator
from entity_linking.matching.util import MatchingScenario
from entity_linking.matching.io import load_candidate_alignment


class Matcher(ABC):
    MODE_TRAIN, MODE_EVAL, MODE_TEST = 'train', 'eval', 'test'

    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__()
        self.scenario = scenario
        self.id = params['id']

    def get_approach_name(self) -> str:
        param_dict = self._get_param_dict()
        if 'bm' in param_dict and param_dict['bm'].startswith('sentence-transformers/'):
            param_dict['bm'] = param_dict['bm'][len('sentence-transformers/'):]
        approach_params = [self.id] + [f'{k}={v}' for k, v in param_dict.items()]
        return '_'.join(approach_params)

    def _get_param_dict(self) -> dict:
        return {}

    def train(self, train_corpus: DataCorpus, eval_corpus: DataCorpus, eval_on_train: bool) -> Dict[str, CandidateAlignment]:
        utils.get_logger().info('Training matcher..')
        self._train_model(train_corpus, eval_corpus)
        utils.get_logger().info('Making predictions..')
        return {self.MODE_TRAIN: self.predict(self.MODE_TRAIN, train_corpus)} if eval_on_train else {}

    @abstractmethod
    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass

    def test(self, test_corpus: DataCorpus) -> Dict[str, CandidateAlignment]:
        utils.get_logger().info('Testing matcher..')
        return {self.MODE_TEST: self._evaluate(self.MODE_TEST, test_corpus)}

    def _evaluate(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        utils.get_logger().debug('Making predictions..')
        pred_start = datetime.now()
        prediction = self.predict(eval_mode, data_corpus)
        prediction_time_in_seconds = (datetime.now() - pred_start).seconds
        utils.get_logger().debug(f'Running evaluation..')
        evaluator = PrecisionRecallF1Evaluator(self.get_approach_name())
        evaluator.compute_and_log_metrics(eval_mode, prediction, data_corpus.alignment, prediction_time_in_seconds)
        return prediction

    @abstractmethod
    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        pass


class MatcherWithCandidates(Matcher, ABC):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        if params['mm_approach']:
            self.mm_approach = params['mm_approach']
            self.mm_ca = load_candidate_alignment(self.mm_approach)
        if params['me_approach']:
            self.me_approach = params['me_approach']
            single_approach = params['mm_approach'] == self.me_approach
            self.me_ca = self.mm_ca if single_approach else load_candidate_alignment(self.me_approach)

    def _get_param_dict(self) -> dict:
        params = {}
        if hasattr(self, 'mm_approach'):
            params['mma'] = self.mm_approach
        if hasattr(self, 'me_approach'):
            params['mea'] = self.me_approach
        return super()._get_param_dict() | params
