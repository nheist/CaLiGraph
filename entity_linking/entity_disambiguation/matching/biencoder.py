from typing import Set, List, Optional
from sentence_transformers import util as st_util
from sentence_transformers import SentenceTransformer
import utils
from impl.caligraph.entity import ClgEntity
from impl.wikipedia.page_parser import WikiListing
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

    def _get_param_dict(self) -> dict:
        params = {
            'bm': self.base_model,
            'k': self.top_k,
            'apc': self.add_page_context,
            'ale': self.add_listing_entities,
            'aea': self.add_entity_abstract,
            'aki': self.add_kg_info,
            'l': self.loss,
            'bs': self.batch_size,
            'e': self.epochs,
            'ws': self.warmup_steps,
        }
        return super()._get_param_dict() | params

    def _train_model(self, training_set: DataCorpus, eval_set: DataCorpus):
        if self.epochs == 0:
            return  # skip training
        utils.get_logger().debug('Preparing training data..')
        train_dataloader = transformer_util.generate_training_data(training_set, set(), self.batch_size, self.add_page_context, self.add_listing_entities, self.add_entity_abstract, self.add_kg_info)
        train_loss = transformer_util.get_loss_function(self.loss, self.model)
        utils.release_gpu()
        utils.get_logger().debug('Starting training..')
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=self.epochs, warmup_steps=self.warmup_steps, save_best_model=False)

    # HINT: use ANN search with e.g. hnswlib (https://github.com/nmslib/hnswlib/) if exact NN search is too costly
    # EXAMPLE: https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/semantic-search/semantic_search_quora_hnswlib.py
    def predict(self, prefix: str, source: List[WikiListing], target: Optional[List[ClgEntity]]) -> Set[Pair]:
        source_ids_with_input = transformer_util.prepare_listing_items(source, self.add_page_context, self.add_listing_entities)
        source_ids, source_input = list(source_ids_with_input), list(source_ids_with_input.values())
        source_embeddings = self.model.encode(source_input, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        if self.scenario == MatchingScenario.MENTION_MENTION:
            alignment = {(i, j) for _, i, j in st_util.paraphrase_mining_embeddings(source_embeddings, max_pairs=int(5e6), top_k=self.top_k, score_function=st_util.dot_score)}
            alignment_indices = {tuple(sorted([source_ids[i], source_ids[j]])) for i, j in alignment}
        else:  # scenario: MENTION_ENTITY
            target_ids_with_input = transformer_util.prepare_entities(target, self.add_entity_abstract, self.add_kg_info)
            target_ids, target_input = list(target_ids_with_input), list(target_ids_with_input.values())
            target_embeddings = self.model.encode(target_input, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
            matched_pairs = st_util.semantic_search(source_embeddings, target_embeddings, top_k=self.top_k, score_function=st_util.dot_score)
            alignment_indices = {(source_ids[s], target_ids[t['corpus_id']]) for s, ts in enumerate(matched_pairs) for t in ts}
        return {Pair(*item_pair) for item_pair in alignment_indices}
