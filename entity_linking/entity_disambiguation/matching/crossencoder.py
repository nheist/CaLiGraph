from typing import Set, Dict, Union
from collections import defaultdict
import os
import random
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder
import utils
from impl.wikipedia import MentionId
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.util import MatchingScenario, get_model_path
from entity_linking.entity_disambiguation.matching.matcher import MatcherWithCandidates
from entity_linking.entity_disambiguation.matching import transformer_util


class CrossEncoderMatcher(MatcherWithCandidates):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.base_model = params['base_model']
        self.mm_threshold = params['mm_threshold']
        self.me_threshold = params['me_threshold']
        self.add_page_context = params['add_page_context']
        self.add_text_context = params['add_text_context']
        self.add_entity_abstract = params['add_entity_abstract']
        self.add_kg_info = params['add_kg_info']
        # training params
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.warmup_steps = params['warmup_steps']

    def _get_param_dict(self) -> dict:
        params = {
            'bm': self.base_model,
            'mmt': self.mm_threshold,
            'met': self.me_threshold,
            'apc': self.add_page_context,
            'atc': self.add_text_context,
            'aea': self.add_entity_abstract,
            'aki': self.add_kg_info,
            'bs': self.batch_size,
            'e': self.epochs,
            'ws': self.warmup_steps,
        }
        return super()._get_param_dict() | params

    def train(self, train_corpus: DataCorpus, eval_corpus: DataCorpus, save_alignment: bool) -> Dict[str, Set[Pair]]:
        utils.get_logger().info('Training matcher..')
        self._train_model(train_corpus, eval_corpus)
        return {}  # never predict on the training set with the cross-encoder

    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        path_to_model = get_model_path(self.base_model)
        if os.path.exists(path_to_model):  # load local model
            self.model = CrossEncoder(path_to_model, num_labels=1)
            return
        else:  # initialize model from huggingface hub
            self.model = CrossEncoder(self.base_model, num_labels=1)
            transformer_util.add_special_tokens(self.model)
        if self.epochs == 0:
            return  # skip training
        utils.get_logger().debug('Preparing training data..')
        training_examples = []
        if self.scenario.is_MM():
            negatives = [cand for cand in self.mm_candidates[self.MODE_TRAIN] if cand not in train_corpus.alignment]
            negatives_sample = random.sample(negatives, min(len(negatives), train_corpus.alignment.sample_size * 2))
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_MENTION, train_corpus, negatives_sample, self.add_page_context, self.add_text_context, self.add_entity_abstract, self.add_kg_info)
        if self.scenario.is_ME():
            negatives = [cand for cand in self.me_candidates[self.MODE_TRAIN] if cand not in train_corpus.alignment]
            negatives_sample = random.sample(negatives, min(len(negatives), train_corpus.alignment.sample_size * 2))
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_ENTITY, train_corpus, negatives_sample, self.add_page_context, self.add_text_context, self.add_entity_abstract, self.add_kg_info)
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=self.batch_size)
        utils.get_logger().debug('Starting training..')
        utils.release_gpu()
        self.model.fit(train_dataloader=train_dataloader, epochs=self.epochs, warmup_steps=self.warmup_steps, save_best_model=False)
        self.model.save(get_model_path(self.id))

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> Set[Pair]:
        mention_input, _ = data_corpus.get_mention_input(self.add_page_context, self.add_text_context)
        alignment = set()
        if self.scenario.is_MM():
            mm_pairs = self._score_candidates(self.mm_candidates[eval_mode], mention_input, mention_input)
            # take all matches that are higher than threshold
            alignment.update({p for p in mm_pairs if p.confidence > self.mm_threshold})
        if self.scenario.is_ME():
            entity_input = data_corpus.get_entity_input(self.add_entity_abstract, self.add_kg_info)
            me_pairs = self._score_candidates(self.me_candidates[eval_mode], mention_input, entity_input)
            # take only the most likely match for an item (if higher than threshold)
            pairs_by_mention = defaultdict(set)
            for pair in me_pairs:
                if pair.confidence <= self.me_threshold:
                    continue
                pairs_by_mention[pair.source].add(pair)
            alignment.update({max(mention_pairs, key=lambda p: p.confidence) for mention_pairs in pairs_by_mention.values()})
        return alignment

    def _score_candidates(self, candidates: Set[Pair], source_input: Dict[MentionId, str], target_input: Dict[Union[MentionId, int], str]) -> Set[Pair]:
        model_input = [[source_input[s_id], target_input[t_id]] for s_id, t_id, _ in candidates]
        utils.release_gpu()
        predictions = self.model.predict(model_input, batch_size=self.batch_size, show_progress_bar=True)
        return {Pair(cand[0], cand[1], score) for cand, score in zip(candidates, predictions)}
