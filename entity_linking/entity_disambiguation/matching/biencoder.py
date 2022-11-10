from typing import List, Union
import os
import numpy as np
from torch import Tensor, nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
import utils
from entity_linking.entity_disambiguation.data import CandidateAlignment, DataCorpus
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from entity_linking.entity_disambiguation.matching.io import get_model_path
from entity_linking.entity_disambiguation.matching.matcher import Matcher
from entity_linking.entity_disambiguation.matching.lexical import ExactMatcher
from entity_linking.entity_disambiguation.matching import transformer_util


class BiEncoderMatcher(Matcher):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.base_model = params['base_model']
        self.top_k = params['top_k']
        self.ans = params['approximate_neighbor_search']
        self.add_page_context = params['add_page_context']
        self.add_text_context = params['add_text_context']
        self.add_entity_abstract = params['add_entity_abstract']
        self.add_kg_info = params['add_kg_info']
        self.init_exact = params['init_exact']
        # training params
        self.loss = params['loss']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.warmup_steps = params['warmup_steps']
        # cache for entity ids and embeddings (as this is the same for all datasets and samples)
        self.entity_ids = None
        self.entity_embeddings = None

    def _get_param_dict(self) -> dict:
        params = {
            'bm': self.base_model,
            'k': self.top_k,
            'ans': self.ans,
            'apc': self.add_page_context,
            'atc': self.add_text_context,
            'aea': self.add_entity_abstract,
            'aki': self.add_kg_info,
            'ie': self.init_exact,
            'l': self.loss,
            'bs': self.batch_size,
            'e': self.epochs,
            'ws': self.warmup_steps,
        }
        return super()._get_param_dict() | params

    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        path_to_model = get_model_path(self.base_model)
        if os.path.exists(path_to_model):  # load local model
            self.model = SentenceTransformer(path_to_model)
            return
        else:  # initialize model from huggingface hub
            self.model = SentenceTransformer(self.base_model)
            transformer_util.add_special_tokens(self.model)
        if self.epochs == 0:
            return  # skip training
        training_examples = []
        if self.scenario.is_MM():
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_MENTION, train_corpus, [], self.add_page_context, self.add_text_context, self.add_entity_abstract, self.add_kg_info)
        if self.scenario.is_ME():
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_ENTITY, train_corpus, [], self.add_page_context, self.add_text_context, self.add_entity_abstract, self.add_kg_info)
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=self.batch_size)
        utils.get_logger().debug('Training bi-encoder model..')
        utils.release_gpu()
        self.model.fit(train_objectives=[(train_dataloader, self._get_loss_function())], epochs=self.epochs, warmup_steps=self.warmup_steps, save_best_model=False)
        self.model.save(get_model_path(self.id))

    def _get_loss_function(self) -> nn.Module:
        if self.loss == 'COS':
            return losses.CosineSimilarityLoss(model=self.model)
        elif self.loss == 'RL':
            return losses.MultipleNegativesRankingLoss(model=self.model)
        elif self.loss == 'SRL':
            return losses.MultipleNegativesSymmetricRankingLoss(model=self.model)
        raise ValueError(f'Unknown loss identifier: {self.loss}')

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        mention_input, mention_known = data_corpus.get_mention_input(self.add_page_context, self.add_text_context)
        mention_ids = list(mention_input)
        mention_embeddings = self._compute_embeddings(list(mention_input.values()))
        ca = ExactMatcher(self.scenario, {'id': None}).predict(eval_mode, data_corpus) if self.init_exact else CandidateAlignment()
        if self.scenario.is_MM():
            known_mask = [mention_known[m_id] for m_id in mention_ids]
            known_mention_ids = [m_id for m_id, known in zip(mention_ids, known_mask) if known]
            known_mention_embeddings = mention_embeddings[known_mask]
            max_pairs = ca.get_mm_candidate_count() * 2 if self.init_exact else data_corpus.alignment.get_mm_match_count() * 50
            if self.ans:
                transformer_util.approximate_paraphrase_mining_embeddings(ca, known_mention_embeddings, known_mention_ids, max_pairs=max_pairs, top_k=50, add_best=True)
            else:
                transformer_util.paraphrase_mining_embeddings(ca, known_mention_embeddings, known_mention_ids, max_pairs=max_pairs, top_k=50, add_best=True)
        if self.scenario.is_ME():
            if self.entity_embeddings is None:  # init cached target embeddings
                entity_input = data_corpus.get_entity_input(self.add_entity_abstract, self.add_kg_info)
                self.entity_ids = list(entity_input)
                self.entity_embeddings = self._compute_embeddings(list(entity_input.values()))
            if self.ans:
                transformer_util.approximate_semantic_search(ca, mention_embeddings, self.entity_embeddings, mention_ids, self.entity_ids, top_k=self.top_k)
            else:
                transformer_util.semantic_search(ca, mention_embeddings, self.entity_embeddings, mention_ids, self.entity_ids, top_k=self.top_k)
        return ca

    def _compute_embeddings(self, inputs: List[str]) -> Union[Tensor, np.ndarray]:
        utils.get_logger().debug('Computing embeddings..')
        utils.release_gpu()
        return self.model.encode(inputs, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=self.ans, convert_to_tensor=(not self.ans), show_progress_bar=True)
