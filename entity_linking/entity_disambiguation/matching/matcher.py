from typing import Set, List, Optional, Dict
from abc import ABC, abstractmethod
from enum import Enum
from time import process_time
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.evaluation import PrecisionRecallF1Evaluator
from impl.wikipedia.page_parser import WikiListing
from impl.caligraph.entity import ClgEntity


class MatchingScenario(Enum):
    MENTION_MENTION = 'MM'
    MENTION_ENTITY = 'ME'


class MatchingApproach(Enum):
    EXACT = 'exact'
    WORD = 'word'
    POPULARITY = 'popularity'
    BIENCODER = 'biencoder'
    CROSSENCODER = 'crossencoder'


class Matcher(ABC):
    MODE_TRAIN, MODE_EVAL, MODE_TEST = 'train', 'eval', 'test'

    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__()
        self.scenario = scenario
        self.version = params['version']

    def get_approach_id(self) -> str:
        return '_'.join([self.scenario.value, f'v{self.version}', type(self).__name__])

    def train(self, training_set: DataCorpus, eval_set: DataCorpus) -> Dict[str, Set[Pair]]:
        self._train_model(training_set, eval_set)
        return {}
#            self.MODE_TRAIN: self._evaluate(self.MODE_TRAIN, training_set),
#            self.MODE_EVAL: self._evaluate(self.MODE_EVAL, eval_set)
#        }

    @abstractmethod
    def _train_model(self, training_set: DataCorpus, eval_set: DataCorpus):
        pass

    def test(self, test_set: DataCorpus) -> Dict[str, Set[Pair]]:
        return {self.MODE_TEST: self._evaluate(self.MODE_TEST, test_set)}

    def _evaluate(self, prefix: str, data_corpus: DataCorpus) -> Set[Pair]:
        pred_start = process_time()
        prediction = self.predict(prefix, data_corpus.source, data_corpus.target)
        prediction_time_in_seconds = int(process_time() - pred_start)

        evaluator = PrecisionRecallF1Evaluator(self.get_approach_id())
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

    def get_approach_id(self) -> str:
        return f'{super().get_approach_id()}_b={self.blocking_approach}'
