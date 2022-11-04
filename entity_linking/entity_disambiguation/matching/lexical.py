from typing import Set, List
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
from tqdm import tqdm
from unidecode import unidecode
from nltk.corpus import stopwords
from fastbm25 import fastbm25
from impl.util.string import make_alphanumeric
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from entity_linking.entity_disambiguation.matching.matcher import Matcher


STOPWORDS = set(stopwords.words('english'))


class LexicalMatcher(Matcher, ABC):
    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> Set[Pair]:
        alignment = set()
        if self.scenario.is_MM():
            mention_grouping = self._make_grouping(data_corpus.get_mention_labels(True))
            for mention_group in mention_grouping.values():
                alignment.update({Pair(*sorted(mention_pair), 1) for mention_pair in itertools.combinations(mention_group, 2)})
        if self.scenario.is_ME():
            mention_grouping = self._make_grouping(data_corpus.get_mention_labels())
            entity_grouping = self._make_grouping({res.idx: res.get_label() for res in data_corpus.get_entities()})
            for key in set(mention_grouping).intersection(set(entity_grouping)):
                mention_group, entity_group = mention_grouping[key], entity_grouping[key]
                alignment.update({Pair(*pair, 1) for pair in itertools.product(mention_group, entity_group)})
        return alignment

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

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> Set[Pair]:
        alignment = set()
        if self.scenario.is_MM():
            tokenized_mentions = {m_id: _tokenize_label(label) for m_id, label in data_corpus.get_mention_labels(True).items()}
            mention_ids = list(tokenized_mentions)
            model = fastbm25(list(tokenized_mentions.values()))
            for m_id, tokens in tqdm(tokenized_mentions.items(), desc='BM25/MM'):
                alignment.update({Pair(*sorted([m_id, mention_ids[idx]]), score) for _, idx, score in model.top_k_sentence(tokens, k=self.top_k)})
        if self.scenario.is_ME():
            tokenized_mentions = {m_id: _tokenize_label(label) for m_id, label in data_corpus.get_mention_labels().items()}
            tokenized_ents = {e.idx: _tokenize_label(e.get_label()) for e in data_corpus.get_entities()}
            ent_ids = list(tokenized_ents)
            model = fastbm25(list(tokenized_ents.values()))
            for m_id, tokens in tqdm(tokenized_mentions.items(), desc='BM25/ME'):
                alignment.update({Pair(m_id, ent_ids[idx], score) for _, idx, score in model.top_k_sentence(tokens, k=self.top_k)})
        return alignment


def _tokenize_label(label: str) -> List[str]:
    words = make_alphanumeric(unidecode(label.lower())).split()
    filtered_words = [w for w in words if w not in STOPWORDS]
    return filtered_words or words
