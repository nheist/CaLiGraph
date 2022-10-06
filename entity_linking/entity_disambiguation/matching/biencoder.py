from typing import Set, List, Optional
from collections import defaultdict
from sentence_transformers import util as st_util
from sentence_transformers import SentenceTransformer
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.matcher import Matcher, MatchingScenario
from entity_linking.entity_disambiguation.matching import transformer_util
from impl.caligraph.entity import ClgEntity
from impl.wikipedia.page_parser import WikiListing


class BiEncoderMatcher(Matcher):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        self.base_model = params['base_model']
        self.loss = params['loss']
        self.epochs = params['epochs']
        self.warmup_steps = params['warmup_steps']
        self.batch_size = params['batch_size']
        self.top_k = params['top_k']
        # prepare Bi-Encoder
        self.model = SentenceTransformer(self.base_model)
        transformer_util.add_special_tokens(self.model)

    def get_approach_id(self) -> str:
        return super().get_approach_id() + f'_bm={self.base_model}_k={self.top_k}_l={self.loss}_bs={self.batch_size}_e={self.epochs}_ws={self.warmup_steps}'

    def _train_model(self, training_set: DataCorpus, eval_set: DataCorpus):
        if self.epochs == 0:
            return  # skip training
        train_dataloader = transformer_util.generate_training_data(training_set, self.batch_size)
        train_loss = transformer_util.get_loss_function(self.loss, self.model)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=self.epochs, warmup_steps=self.warmup_steps, save_best_model=False)

    # HINT: use ANN search with e.g. hnswlib (https://github.com/nmslib/hnswlib/) if exact NN search is too costly
    # EXAMPLE: https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/semantic-search/semantic_search_quora_hnswlib.py
    def predict(self, prefix: str, source: List[WikiListing], target: Optional[List[ClgEntity]]) -> Set[Pair]:
        source_ids_with_input = transformer_util.prepare_listing_items(source)
        source_ids, source_input = list(source_ids_with_input), list(source_ids_with_input.values())
        source_embeddings = self.model.encode(source_input, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        if self.scenario == MatchingScenario.MENTION_MENTION:
            alignment = defaultdict(set)
            for score, i, j in st_util.paraphrase_mining_embeddings(source_embeddings, max_pairs=10**8, top_k=self.top_k, score_function=st_util.dot_score):
                alignment[i].add((j, score))
            alignment = {(i, j[0]) for i, js in alignment.items() for j in sorted(js, key=lambda x: x[1], reverse=True)[:self.top_k]}
            alignment_indices = {tuple(sorted([source_ids[i], source_ids[j]])) for i, j in alignment}
        else:  # scenario: MENTION_ENTITY
            target_ids_with_input = transformer_util.prepare_entities(target)
            target_ids, target_input = list(target_ids_with_input), list(target_ids_with_input.values())
            target_embeddings = self.model.encode(target_input, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
            matched_pairs = st_util.semantic_search(source_embeddings, target_embeddings, top_k=self.top_k, score_function=st_util.dot_score)
            alignment_indices = {(source_ids[s], target_ids[t['corpus_id']]) for s, ts in matched_pairs.items() for t in ts}
        return {Pair(*item_pair) for item_pair in alignment_indices}
