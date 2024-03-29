from typing import List, Union
import os
import numpy as np
from torch import Tensor, nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
import utils
from impl.subject_entity.entity_disambiguation.data import CandidateAlignment, DataCorpus
from impl.subject_entity.entity_disambiguation.matching.util import MatchingScenario
from impl.subject_entity.entity_disambiguation.matching.io import get_cache_path
from impl.subject_entity.entity_disambiguation.matching.matcher import Matcher
from impl.subject_entity.entity_disambiguation.matching import transformer_util


class BiEncoderMatcher(Matcher):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.base_model = params['base_model']
        self.train_sample = params['train_sample']
        self.top_k = params['top_k']
        self.ans = params['approximate_neighbor_search']
        self.add_page_context = params['add_page_context']
        self.add_text_context = params['add_text_context']
        self.add_entity_abstract = params['add_entity_abstract']
        self.add_kg_info = params['add_kg_info']
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
            'ts': self.train_sample,
            'k': self.top_k,
            'ans': self.ans,
            'apc': self.add_page_context,
            'atc': self.add_text_context,
            'aea': self.add_entity_abstract,
            'aki': self.add_kg_info,
            'l': self.loss,
            'bs': self.batch_size,
            'e': self.epochs,
            'ws': self.warmup_steps,
        }
        return super()._get_param_dict() | params

    def _train_model(self, train_corpus: DataCorpus):
        # load local model, if available
        path_to_model = get_cache_path(self.id)
        if os.path.exists(path_to_model):
            self.model = SentenceTransformer(path_to_model)
            return
        # initialize model from huggingface hub
        self.model = SentenceTransformer(self.base_model)
        transformer_util.add_special_tokens(self.model)
        if self.epochs == 0:
            return  # skip training
        training_examples = []
        if self.scenario.is_mention_mention():
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_MENTION, train_corpus, self.train_sample, [], self.add_page_context, self.add_text_context, self.add_entity_abstract, self.add_kg_info)
        if self.scenario.is_mention_entity():
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_ENTITY, train_corpus, self.train_sample, [], self.add_page_context, self.add_text_context, self.add_entity_abstract, self.add_kg_info)
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=self.batch_size)
        utils.get_logger().debug('Training bi-encoder model..')
        utils.release_gpu()
        self.model.fit(train_objectives=[(train_dataloader, self._get_loss_function())], epochs=self.epochs, warmup_steps=self.warmup_steps, save_best_model=False)
        utils.get_logger().debug(f'Storing model {self.id}..')
        self.model.save(get_cache_path(self.id))

    def _get_loss_function(self) -> nn.Module:
        if self.loss == 'COS':
            return losses.CosineSimilarityLoss(model=self.model)
        elif self.loss == 'RL':
            return losses.MultipleNegativesRankingLoss(model=self.model)
        elif self.loss == 'SRL':
            return losses.MultipleNegativesSymmetricRankingLoss(model=self.model)
        raise ValueError(f'Unknown loss identifier: {self.loss}')

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> CandidateAlignment:
        mention_input = data_corpus.get_mention_input(self.add_page_context, self.add_text_context)
        mention_ids = list(mention_input)
        # compute embeddings
        mention_embeddings = self._compute_embeddings(list(mention_input.values()))
        if self.scenario.is_mention_entity():
            if self.entity_embeddings is None:  # init cached target embeddings
                entity_input = data_corpus.get_entity_input(self.add_entity_abstract, self.add_kg_info)
                self.entity_ids = list(entity_input)
                self.entity_embeddings = self._compute_embeddings(list(entity_input.values()))
        # find nearest neighbors
        ca = CandidateAlignment()
        if self.scenario.is_mention_mention():
            if self.ans:
                transformer_util.approximate_semantic_search(ca, mention_embeddings, mention_embeddings, mention_ids, mention_ids, top_k=self.top_k + 1)
            else:
                transformer_util.semantic_search(ca, mention_embeddings, mention_embeddings, mention_ids, mention_ids, top_k=self.top_k + 1)
        if self.scenario.is_mention_entity():
            if self.ans:
                transformer_util.approximate_semantic_search(ca, mention_embeddings, self.entity_embeddings, mention_ids, self.entity_ids, top_k=self.top_k)
            else:
                transformer_util.semantic_search(ca, mention_embeddings, self.entity_embeddings, mention_ids, self.entity_ids, top_k=self.top_k)
        return ca

    def _compute_embeddings(self, inputs: List[str]) -> Union[Tensor, np.ndarray]:
        utils.get_logger().debug('Computing embeddings..')
        utils.release_gpu()
        return self.model.encode(inputs, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=self.ans, convert_to_tensor=(not self.ans), show_progress_bar=True)
