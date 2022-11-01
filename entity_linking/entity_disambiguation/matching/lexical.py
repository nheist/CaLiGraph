from typing import Set
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
from unidecode import unidecode
from nltk.corpus import stopwords
from impl.util.string import make_alphanumeric
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.matcher import Matcher


class LexicalMatcher(Matcher, ABC):
    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> Set[Pair]:
        mention_grouping = self._make_grouping(data_corpus.get_mention_labels())
        alignment = set()
        if self.scenario.is_MM():
            for mention_group in mention_grouping.values():
                alignment.update({Pair(*sorted(mention_pair), 1) for mention_pair in itertools.combinations(mention_group, 2)})
        if self.scenario.is_ME():
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
        english_stopwords = set(stopwords.words('english'))
        grouping = defaultdict(set)
        for item_id, label in item_labels.items():
            words = set(make_alphanumeric(unidecode(label.lower())).split())
            filtered_words = words.difference(english_stopwords)
            group_key = tuple(sorted(filtered_words or words))
            grouping[group_key].add(item_id)
        return grouping
