from typing import Set, List, Optional
from collections import defaultdict
from sentence_transformers import CrossEncoder
import utils
from impl.caligraph.entity import ClgEntity
from impl.wikipedia.page_parser import WikiListing
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.matcher import MatcherWithCandidates, MatchingScenario
from entity_linking.entity_disambiguation.matching import transformer_util


class CrossEncoderMatcher(MatcherWithCandidates):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        self.base_model = params['base_model']
        self.loss = params['loss']
        self.epochs = params['epochs']
        self.warmup_steps = params['warmup_steps']
        self.batch_size = params['batch_size']
        # prepare Cross-Encoder
        self.model = CrossEncoder(self.base_model, num_labels=1)
        transformer_util.add_special_tokens(self.model)

    def _train_model(self, training_set: DataCorpus, eval_set: DataCorpus):
        if self.epochs == 0:
            return  # skip training
        train_dataloader = transformer_util.generate_training_data(training_set, self.batch_size)
        train_loss = transformer_util.get_loss_function(self.loss, self.model)
        utils.release_gpu()
        self.model.fit(train_dataloader=train_dataloader, loss_fct=train_loss, epochs=self.epochs, warmup_steps=self.warmup_steps, save_best_model=False)

    def predict(self, prefix: str, source: List[WikiListing], target: Optional[List[ClgEntity]]) -> Set[Pair]:
        source_input = transformer_util.prepare_listing_items(source)
        if self.scenario == MatchingScenario.MENTION_MENTION:
            target_input = source_input
        else:
            target_input = transformer_util.prepare_entities(target)
        candidates = self.candidates[prefix]
        model_input = [[source_input[source_id], target_input[target_id]] for source_id, target_id in candidates]
        candidate_scores = self.model.predict(model_input, batch_size=self.batch_size, show_progress_bar=True)
        if self.scenario == MatchingScenario.MENTION_MENTION:
            # take all matches that are higher than threshold
            alignment = [cand for cand, score in zip(candidates, candidate_scores) if score > .5]
        else:  # MENTION_ENTITY
            # take only the most likely match for an item
            item_entity_scores = defaultdict(set)
            for (item_id, entity_id), score in zip(candidates, candidate_scores):
                item_entity_scores[item_id].add((entity_id, score))
            alignment = [(i, max(js, key=lambda x: x[1])[0]) for i, js in item_entity_scores.items()]
        return {Pair(*item_pair) for item_pair in alignment}
