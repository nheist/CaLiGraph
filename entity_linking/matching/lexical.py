from typing import List
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
from tqdm import tqdm
from unidecode import unidecode
from nltk.corpus import stopwords
from fastbm25 import fastbm25
from impl.util.string import make_alphanumeric
from entity_linking.data import CandidateAlignment, DataCorpus
from entity_linking.matching.util import MatchingScenario
from entity_linking.matching.matcher import Matcher


STOPWORDS = set(stopwords.words('english'))


class LexicalMatcher(Matcher, ABC):
    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        ca = CandidateAlignment(self.scenario)
        if self.scenario.is_MM():
            mention_grouping = self._make_grouping(data_corpus.get_mention_labels(True))
            for mention_group in mention_grouping.values():
                for mention_pair in itertools.combinations(mention_group, 2):
                    ca.add_candidate(mention_pair, 1)
        if self.scenario.is_ME():
            mention_grouping = self._make_grouping(data_corpus.get_mention_labels())
            entity_grouping = self._make_grouping({res.idx: res.get_label() for res in data_corpus.get_entities()})
            for key in set(mention_grouping).intersection(set(entity_grouping)):
                mention_group, entity_group = mention_grouping[key], entity_grouping[key]
                for pair in itertools.product(mention_group, entity_group):
                    ca.add_candidate(pair, 1)
        return ca

    @abstractmethod
    def _make_grouping(self, item_labels: dict) -> dict:
        pass


class ExactMatcher(LexicalMatcher):
    def _make_grouping(self, item_labels: dict) -> dict:
        grouping = defaultdict(set)
        for item_id, label in item_labels.items():
            group_key = make_alphanumeric(unidecode(label.lower()))
            grouping[group_key].add(item_id)
        return grouping


class WordMatcher(LexicalMatcher):
    def _make_grouping(self, item_labels: dict) -> dict:
        grouping = defaultdict(set)
        for item_id, label in item_labels.items():
            group_key = tuple(sorted(set(_tokenize_label(label))))
            grouping[group_key].add(item_id)
        return grouping


class BM25Matcher(Matcher):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.top_k = params['top_k']

    def _get_param_dict(self) -> dict:
        return super()._get_param_dict() | {'k': self.top_k}

    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        ca = CandidateAlignment(self.scenario)
        if self.scenario.is_MM():
            tokenized_mentions = {m_id: _tokenize_label(label) for m_id, label in data_corpus.get_mention_labels(True).items()}
            mention_ids = list(tokenized_mentions)
            model = fastbm25(list(tokenized_mentions.values()))
            for m_id, tokens in tqdm(tokenized_mentions.items(), desc='BM25/MM'):
                for _, idx, score in model.top_k_sentence(tokens, k=self.top_k + 1):
                    ca.add_candidate((m_id, mention_ids[idx]), score)
        if self.scenario.is_ME():
            tokenized_mentions = {m_id: _tokenize_label(label) for m_id, label in data_corpus.get_mention_labels().items()}
            tokenized_ents = {e.idx: _tokenize_label(e.get_label()) for e in data_corpus.get_entities()}
            ent_ids = list(tokenized_ents)
            model = fastbm25(list(tokenized_ents.values()))
            for m_id, tokens in tqdm(tokenized_mentions.items(), desc='BM25/ME'):
                for _, idx, score in model.top_k_sentence(tokens, k=self.top_k):
                    ca.add_candidate((m_id, ent_ids[idx]), score)
        return ca


def _tokenize_label(label: str) -> List[str]:
    words = make_alphanumeric(unidecode(label.lower())).split()
    filtered_words = [w for w in words if w not in STOPWORDS]
    return filtered_words or words
