from typing import Set, List, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
from unidecode import unidecode
from nltk.corpus import stopwords
from impl.util.string import make_alphanumeric
from impl.wikipedia.page_parser import WikiListing
from impl.caligraph.entity import ClgEntity
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.matcher import Matcher


class LexicalMatcher(Matcher, ABC):
    def _train_model(self, training_set: DataCorpus, eval_set: DataCorpus):
        pass  # no training necessary

    def predict(self, prefix: str, source: List[WikiListing], target: Optional[List[ClgEntity]]) -> Set[Pair]:
        alignment = set()
        source_grouping = self._make_grouping(self._get_item_labels_for_listings(source))
        if target is None:
            for item_group in source_grouping.values():
                alignment.update({Pair(*sorted(item_pair)) for item_pair in itertools.combinations(item_group, 2)})
        else:
            target_grouping = self._make_grouping({res.idx: res.get_label() for res in target})
            for key in set(source_grouping).intersection(set(target_grouping)):
                source_group, target_group = source_grouping[key], target_grouping[key]
                alignment.update({Pair(*item_pair) for item_pair in itertools.product(source_group, target_group)})
        return alignment

    @classmethod
    def _get_item_labels_for_listings(cls, listings: List[WikiListing]) -> dict:
        item_labels = {}
        for listing in listings:
            for item in listing.get_items():
                if item.subject_entity is None:
                    continue
                item_labels[(listing.page_idx, listing.idx, item.idx)] = item.subject_entity.label
        return item_labels

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
