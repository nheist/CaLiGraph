from typing import Set, Dict, Union, Tuple, List
from collections import defaultdict
import os
import random
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder
import utils
from impl.wikipedia import MentionId
from impl.subject_entity.entity_disambiguation.data import CandidateAlignment, DataCorpus, Pair
from impl.subject_entity.entity_disambiguation.matching.util import MatchingScenario
from impl.subject_entity.entity_disambiguation.matching.io import get_cache_path
from impl.subject_entity.entity_disambiguation.matching.matcher import MatcherWithCandidates
from impl.subject_entity.entity_disambiguation.matching import transformer_util


class CrossEncoderMatcher(MatcherWithCandidates):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.base_model = params['base_model']
        self.train_sample = params['train_sample']
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
            'ts': self.train_sample,
            'apc': self.add_page_context,
            'atc': self.add_text_context,
            'aea': self.add_entity_abstract,
            'aki': self.add_kg_info,
            'bs': self.batch_size,
            'e': self.epochs,
            'ws': self.warmup_steps,
        }
        return super()._get_param_dict() | params

    def train(self, train_corpus: DataCorpus, save_alignment: bool) -> Dict[str, CandidateAlignment]:
        utils.get_logger().info('Training matcher..')
        self._train_model(train_corpus)
        return {}  # never predict on the training set with the cross-encoder

    def is_model_ready(self) -> bool:
        return os.path.exists(get_cache_path(self.base_model))

    def _train_model(self, train_corpus: DataCorpus):
        path_to_model = get_cache_path(self.base_model)
        if self.is_model_ready():  # load local model
            self.model = CrossEncoder(path_to_model, num_labels=1)
            return
        else:  # initialize model from huggingface hub
            self.model = CrossEncoder(self.base_model, num_labels=1)
            transformer_util.add_special_tokens(self.model)
        if self.epochs == 0:
            return  # skip training
        training_examples = []
        negative_sample_size = 2 * self.train_sample * 10**6
        if self.scenario.is_mention_mention():
            negatives = [pair for pair in self.mm_ca[self.MODE_TRAIN].get_mm_candidates(False) if not train_corpus.alignment.has_match(pair)]
            negatives_sample = random.sample(negatives, min(len(negatives), negative_sample_size))
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_MENTION, train_corpus, self.train_sample, negatives_sample, self.add_page_context, self.add_text_context, self.add_entity_abstract, self.add_kg_info)
        if self.scenario.is_mention_entity():
            negatives = [pair for pair in self.me_ca[self.MODE_TRAIN].get_me_candidates(False) if not train_corpus.alignment.has_match(pair)]
            negatives_sample = random.sample(negatives, min(len(negatives), negative_sample_size))
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_ENTITY, train_corpus, self.train_sample, negatives_sample, self.add_page_context, self.add_text_context, self.add_entity_abstract, self.add_kg_info)
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=self.batch_size)
        utils.get_logger().debug('Training cross-encoder model..')
        utils.release_gpu()
        self.model.fit(train_dataloader=train_dataloader, epochs=self.epochs, warmup_steps=self.warmup_steps, save_best_model=False)
        utils.get_logger().debug(f'Storing model {self.id}..')
        self.model.save(get_cache_path(self.id))

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        mention_input = data_corpus.get_mention_input(self.add_page_context, self.add_text_context)
        ca = CandidateAlignment()
        if self.scenario.is_mention_mention():
            utils.get_logger().debug('Computing mention-mention matches..')
            for pair, score in self._score_pairs(list(self.mm_ca[eval_mode].get_mm_candidates(False)), mention_input, mention_input):
                ca.add_candidate(pair, score)
        if self.scenario.is_mention_entity():
            entity_input = data_corpus.get_entity_input(self.add_entity_abstract, self.add_kg_info)
            utils.get_logger().debug('Computing mention-entity matches..')
            # take only the most likely match for an item (if higher than threshold)
            scored_pairs_by_mention = defaultdict(set)
            for pair, score in self._score_pairs(list(self.me_ca[eval_mode].get_me_candidates(False)), mention_input, entity_input):
                scored_pairs_by_mention[pair[0]].add((pair, score))
            for scored_pairs in scored_pairs_by_mention.values():
                pair, score = max(scored_pairs, key=lambda x: x[1])
                ca.add_candidate(pair, score)
        return ca

    def _score_pairs(self, pairs: List[Pair], source_input: Dict[MentionId, str], target_input: Dict[Union[MentionId, int], str]) -> Set[Tuple[Pair, float]]:
        model_input = [[source_input[s_id], target_input[t_id]] for s_id, t_id in pairs]
        utils.release_gpu()
        predictions = self.model.predict(model_input, batch_size=self.batch_size, show_progress_bar=True)
        return set(zip(pairs, predictions))
