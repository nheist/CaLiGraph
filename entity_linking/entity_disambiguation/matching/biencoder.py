from typing import Set
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
import utils
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from entity_linking.entity_disambiguation.matching.matcher import Matcher
from entity_linking.entity_disambiguation.matching import transformer_util


class BiEncoderMatcher(Matcher):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.base_model = params['base_model']
        self.top_k = params['top_k']
        self.add_page_context = params['add_page_context']
        self.add_category_context = params['add_category_context']
        self.add_listing_entities = params['add_listing_entities']
        self.add_entity_abstract = params['add_entity_abstract']
        self.add_kg_info = params['add_kg_info']
        # training params
        self.loss = params['loss']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.warmup_steps = params['warmup_steps']
        # prepare Bi-Encoder
        self.model = SentenceTransformer(self.base_model)
        transformer_util.add_special_tokens(self.model)
        # cache for target ids and embeddings (as this is the same for all datasets and samples)
        self.target_ids = None
        self.target_embeddings = None

    def _get_param_dict(self) -> dict:
        params = {
            'bm': self.base_model,
            'k': self.top_k,
            'apc': self.add_page_context,
            'acc': self.add_category_context,
            'ale': self.add_listing_entities,
            'aea': self.add_entity_abstract,
            'aki': self.add_kg_info,
            'l': self.loss,
            'bs': self.batch_size,
            'e': self.epochs,
            'ws': self.warmup_steps,
        }
        return super()._get_param_dict() | params

    def _train_model(self, train_corpus: DataCorpus, eval_corpus: DataCorpus):
        if self.epochs == 0:
            return  # skip training
        utils.get_logger().debug('Preparing training data..')
        training_examples = []
        if self.scenario.is_MM():
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_MENTION, train_corpus, [], self.add_page_context, self.add_category_context, self.add_listing_entities, self.add_entity_abstract, self.add_kg_info)
        if self.scenario.is_ME():
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_ENTITY, train_corpus, [], self.add_page_context, self.add_category_context, self.add_listing_entities, self.add_entity_abstract, self.add_kg_info)
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=self.batch_size)
        utils.get_logger().debug('Starting training..')
        utils.release_gpu()
        self.model.fit(train_objectives=[(train_dataloader, self._get_loss_function())], epochs=self.epochs, warmup_steps=self.warmup_steps, save_best_model=False)

    def _get_loss_function(self) -> nn.Module:
        if self.loss == 'COS':
            return losses.CosineSimilarityLoss(model=self.model)
        elif self.loss == 'RL':
            return losses.MultipleNegativesRankingLoss(model=self.model)
        elif self.loss == 'SRL':
            return losses.MultipleNegativesSymmetricRankingLoss(model=self.model)
        raise ValueError(f'Unknown loss identifier: {self.loss}')

    # HINT: use ANN search with e.g. hnswlib (https://github.com/nmslib/hnswlib/) if exact NN search is too costly
    # EXAMPLE: https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/semantic-search/semantic_search_quora_hnswlib.py
    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> Set[Pair]:
        source_ids_with_input, source_known = transformer_util.prepare_listing_items(data_corpus.get_listings(), self.add_page_context, self.add_category_context, self.add_listing_entities)
        source_ids, source_input = list(source_ids_with_input), list(source_ids_with_input.values())
        source_embeddings = self.model.encode(source_input, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        alignment = set()
        if self.scenario.is_MM():
            known_mask = [source_known[m_id] for m_id in source_ids]
            mm_source_ids = [m_id for m_id, known in zip(source_ids, known_mask) if known]
            mm_source_embeddings = source_embeddings[known_mask]
            matched_pairs = transformer_util.paraphrase_mining_embeddings(mm_source_embeddings, mm_source_ids, max_pairs=int(5e6), top_k=50, add_best=True)
            alignment.update(matched_pairs)
        if self.scenario.is_ME():
            if self.target_embeddings is None:  # init cached target embeddings
                target_ids_with_input = transformer_util.prepare_entities(data_corpus.get_entities(), self.add_entity_abstract, self.add_kg_info)
                self.target_ids, target_input = list(target_ids_with_input), list(target_ids_with_input.values())
                self.target_embeddings = self.model.encode(target_input, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
            matched_pairs = transformer_util.semantic_search(source_embeddings, self.target_embeddings, source_ids, self.target_ids, top_k=self.top_k)
            alignment.update(matched_pairs)
        return alignment
