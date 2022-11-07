from typing import Set, List
import os
from torch import Tensor, nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
import utils
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.util import MatchingScenario, get_model_path
from entity_linking.entity_disambiguation.matching.matcher import Matcher
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
        utils.get_logger().debug('Preparing training data..')
        training_examples = []
        if self.scenario.is_MM():
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_MENTION, train_corpus, [], self.add_page_context, self.add_text_context, self.add_entity_abstract, self.add_kg_info)
        if self.scenario.is_ME():
            training_examples += transformer_util.generate_training_data(MatchingScenario.MENTION_ENTITY, train_corpus, [], self.add_page_context, self.add_text_context, self.add_entity_abstract, self.add_kg_info)
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=self.batch_size)
        utils.get_logger().debug('Starting training..')
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

    # HINT: use ANN search with e.g. hnswlib (https://github.com/nmslib/hnswlib/) if exact NN search is too costly
    # EXAMPLE: https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/semantic-search/semantic_search_quora_hnswlib.py
    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> Set[Pair]:
        mention_input, mention_known = data_corpus.get_mention_input(self.add_page_context, self.add_text_context)
        mention_ids = list(mention_input)
        mention_embeddings = self._compute_embeddings(list(mention_input.values()))
        alignment = set()
        if self.scenario.is_MM():
            known_mask = [mention_known[m_id] for m_id in mention_ids]
            known_mention_ids = [m_id for m_id, known in zip(mention_ids, known_mask) if known]
            known_mention_embeddings = mention_embeddings[known_mask]
            max_pairs = data_corpus.alignment.mm_match_count() * 50
            if self.ans:
                matched_pairs = transformer_util.approximate_paraphrase_mining_embeddings(known_mention_embeddings, known_mention_ids, max_pairs=max_pairs, top_k=50, add_best=True)
            else:
                matched_pairs = transformer_util.paraphrase_mining_embeddings(known_mention_embeddings, known_mention_ids, max_pairs=max_pairs, top_k=50, add_best=True)
            alignment.update(matched_pairs)
        if self.scenario.is_ME():
            if self.entity_embeddings is None:  # init cached target embeddings
                entity_input = data_corpus.get_entity_input(self.add_entity_abstract, self.add_kg_info)
                self.entity_ids = list(entity_input)
                self.entity_embeddings = self._compute_embeddings(list(entity_input.values()))
            if self.ans:
                matched_pairs = transformer_util.approximate_semantic_search(mention_embeddings, self.entity_embeddings, mention_ids, self.entity_ids, top_k=self.top_k)
            else:
                matched_pairs = transformer_util.semantic_search(mention_embeddings, self.entity_embeddings, mention_ids, self.entity_ids, top_k=self.top_k)
            alignment.update(matched_pairs)
        return alignment

    def _compute_embeddings(self, inputs: List[str]) -> Tensor:
        return self.model.encode(inputs, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=self.ans, convert_to_tensor=(not self.ans), show_progress_bar=True)
