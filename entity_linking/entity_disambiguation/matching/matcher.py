from typing import Set, Dict
from abc import ABC, abstractmethod
from time import process_time
import utils
from entity_linking.entity_disambiguation.data import Pair, Alignment, DataCorpus
from entity_linking.entity_disambiguation.evaluation import PrecisionRecallF1Evaluator
from entity_linking.entity_disambiguation.matching.util import MatchingScenario, load_candidates
from impl.wikipedia.page_parser import MentionId


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
        pred_start = process_time()
        prediction = self.predict(eval_mode, data_corpus)
        prediction_time_in_seconds = int(process_time() - pred_start)

        prediction_by_scenario = {}
        if self.scenario.is_MM():
            scenario = MatchingScenario.MENTION_MENTION
            prediction_by_scenario[scenario] = self._evaluate_scenario(eval_mode, prediction, data_corpus.alignment, prediction_time_in_seconds, scenario)
        if self.scenario.is_ME():
            scenario = MatchingScenario.MENTION_ENTITY
            prediction_by_scenario[scenario] = self._evaluate_scenario(eval_mode, prediction, data_corpus.alignment, prediction_time_in_seconds, scenario)
        return prediction_by_scenario

    def _evaluate_scenario(self, eval_mode: str, prediction: Set[Pair], alignment: Alignment, prediction_time: int, scenario: MatchingScenario) -> Set[Pair]:
        pred_type = MentionId if scenario == MatchingScenario.MENTION_MENTION else int
        scenario_prediction = {pred for pred in prediction if isinstance(pred.target, pred_type)}
        scenario_prediction_time = int(len(scenario_prediction) / len(prediction) * prediction_time)
        evaluator = PrecisionRecallF1Evaluator(self.get_approach_name(), scenario)
        evaluator.compute_and_log_metrics(eval_mode, scenario_prediction, alignment, scenario_prediction_time)
        return scenario_prediction

    @abstractmethod
    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> Set[Pair]:
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
