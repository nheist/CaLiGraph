from typing import List, Dict
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
import queue
from tqdm import tqdm
from unidecode import unidecode
from nltk.corpus import stopwords
from fastbm25 import fastbm25
from impl.util.string import make_alphanumeric
from impl.wikipedia import MentionId
from entity_linking.entity_disambiguation.data import CandidateAlignment, DataCorpus
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from entity_linking.entity_disambiguation.matching.matcher import Matcher


STOPWORDS = set(stopwords.words('english'))


class LexicalMatcher(Matcher, ABC):
    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        pass  # no training necessary

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        ca = CandidateAlignment()
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
        ca = CandidateAlignment()
        if self.scenario.is_MM():
            tokenized_mentions = {m_id: _tokenize_label(label) for m_id, label in data_corpus.get_mention_labels(True).items()}
            max_pairs = data_corpus.alignment.get_match_count(MatchingScenario.MENTION_MENTION) * 50
            self._find_best_pairs(ca, tokenized_mentions, max_pairs, 50, True)
        if self.scenario.is_ME():
            tokenized_mentions = {m_id: _tokenize_label(label) for m_id, label in data_corpus.get_mention_labels().items()}
            tokenized_ents = {e.idx: _tokenize_label(e.get_label()) for e in data_corpus.get_entities()}
            ent_ids = list(tokenized_ents)
            model = fastbm25(list(tokenized_ents.values()))
            for m_id, tokens in tqdm(tokenized_mentions.items(), desc='BM25/ME'):
                for _, idx, score in model.top_k_sentence(tokens, k=self.top_k):
                    ca.add_candidate((m_id, ent_ids[idx]), score)
        return ca

    @classmethod
    def _find_best_pairs(cls, ca: CandidateAlignment, tokenized_mentions: Dict[MentionId, List[str]], max_pairs: int = 500000, top_k: int = 100, add_best: bool = False):
        top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs
        best_pairs_per_item = defaultdict(lambda: (None, 0.0))
        top_pairs = queue.PriorityQueue()
        min_score = -1
        num_added = 0

        mention_ids = list(tokenized_mentions)
        model = fastbm25(list(tokenized_mentions.values()))
        for mention_a, tokens in tqdm(tokenized_mentions.items(), desc='BM25/MM'):
            for _, idx, score in model.top_k_sentence(tokens, k=top_k):
                mention_b = mention_ids[idx]
                if mention_a == mention_b:
                    continue
                if add_best:
                    # collect best pairs per item
                    if best_pairs_per_item[mention_a][1] < score:
                        best_pairs_per_item[mention_a] = (mention_b, score)
                    if best_pairs_per_item[mention_b][1] < score:
                        best_pairs_per_item[mention_b] = (mention_a, score)
                if score > min_score:
                    # collect overall top pairs
                    top_pairs.put((score, *sorted([mention_a, mention_b])))
                    num_added += 1
                    if num_added >= max_pairs:
                        entry = top_pairs.get()
                        min_score = entry[0]
        # assemble the final pairs
        if add_best:
            for mention_a, (mention_b, score) in best_pairs_per_item.items():
                ca.add_candidate((mention_a, mention_b), score)
        while not top_pairs.empty():
            score, mention_a, mention_b = top_pairs.get()
            ca.add_candidate((mention_a, mention_b), score)


def _tokenize_label(label: str) -> List[str]:
    words = make_alphanumeric(unidecode(label.lower())).split()
    filtered_words = [w for w in words if w not in STOPWORDS]
    return filtered_words or words
