from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import utils
from impl.dbpedia.resource import DbpResourceStore
from impl.wikipedia import WikiPageStore, MentionId
from impl.subject_entity.entity_disambiguation.matching.util import MatchingScenario
from impl.subject_entity.entity_disambiguation.data import get_listing_prediction_corpus, get_data_corpora, CorpusType
from impl.subject_entity.entity_disambiguation.matching.biencoder import BiEncoderMatcher
from impl.subject_entity.entity_disambiguation.matching.crossencoder import CrossEncoderMatcher
from impl.subject_entity.entity_disambiguation.matching.greedy_clustering import NastyLinker


def disambiguate_subject_entities():
    disambiguated_subject_entities = utils.load_or_create_cache('disambiguated_subject_entity_mentions', _disambiguate_subject_entity_mentions)
    WikiPageStore.instance().add_disambiguated_subject_entities(disambiguated_subject_entities)


def _disambiguate_subject_entity_mentions() -> Dict[int, Dict[int, Dict[int, int]]]:
    # compute mention clusters
    biencoder, crossencoder, nastylinker = _init_models()
    mention_clusters = _compute_mention_clusters(biencoder, crossencoder, nastylinker)
    # assign new entity IDs to mention clusters with unknown entity
    disambiguated_subject_entities = defaultdict(lambda: defaultdict(dict))
    current_entity_idx = DbpResourceStore.instance().get_highest_resource_idx()
    for mention_cluster, ent in mention_clusters:
        if ent is None:  # assigning new entity to cluster if it has none
            current_entity_idx += 1
            ent = current_entity_idx
        for mention_id in mention_cluster:
            page_idx, listing_idx, item_idx = mention_id
            disambiguated_subject_entities[page_idx][listing_idx][item_idx] = ent
    return disambiguated_subject_entities


def _init_models() -> Tuple[BiEncoderMatcher, CrossEncoderMatcher, NastyLinker]:
    train_corpus, _, _ = get_data_corpora(CorpusType.LIST, 20)
    biencoder = BiEncoderMatcher(MatchingScenario.FULL, _get_biencoder_config())
    biencoder._train_model(train_corpus)
    crossencoder = CrossEncoderMatcher(MatchingScenario.MENTION_ENTITY, _get_crossencoder_config())
    if not crossencoder.is_model_ready():
        crossencoder.me_ca = {crossencoder.MODE_TRAIN: biencoder.predict(biencoder.MODE_TRAIN, train_corpus)}
    crossencoder._train_model(train_corpus)
    nastylinker = NastyLinker(MatchingScenario.FULL, _get_nastylinker_config())
    return biencoder, crossencoder, nastylinker


def _get_biencoder_config() -> dict:
    be_id = 'biencoder_full'
    return {
        'id': be_id, 'base_model': be_id, 'train_sample': 1, 'top_k': 4, 'approximate_neighbor_search': True,
        'add_page_context': True, 'add_text_context': False, 'add_entity_abstract': True, 'add_kg_info': False,
        'loss': 'SRL', 'batch_size': 384, 'epochs': 1, 'warmup_steps': 0
    }


def _get_crossencoder_config() -> dict:
    ce_id = 'crossencoder_me'
    return {
        'id': ce_id, 'base_model': ce_id, 'train_sample': 1,
        'add_page_context': True, 'add_text_context': False, 'add_entity_abstract': True, 'add_kg_info': False,
        'batch_size': 384, 'epochs': 1, 'warmup_steps': 0
    }


def _get_nastylinker_config() -> dict:
    return {'id': 'nastylinker', 'mm_threshold': .8, 'me_threshold': .95, 'path_threshold': .75}


def _compute_mention_clusters(biencoder: BiEncoderMatcher, crossencoder: CrossEncoderMatcher, nastylinker: NastyLinker) -> List[Tuple[Set[MentionId], Optional[int]]]:
    # make predictions
    corpus = get_listing_prediction_corpus()
    biencoder_ca = biencoder.predict(biencoder.MODE_PREDICT, corpus)
    crossencoder.me_ca = {crossencoder.MODE_PREDICT: biencoder_ca}
    crossencoder_ca = crossencoder.predict(crossencoder.MODE_PREDICT, corpus)
    nastylinker.mm_ca = {nastylinker.MODE_PREDICT: biencoder_ca}
    nastylinker.me_ca = {nastylinker.MODE_PREDICT: crossencoder_ca}
    nastylinker_ca = nastylinker.predict(nastylinker.MODE_PREDICT, corpus)
    # filter out known entities from clustering
    unknown_entity_mention_clusters = []
    for mentions, ent in nastylinker_ca.clustering:
        unknown_mentions = {m for m in mentions if m not in corpus.alignment.mention_to_entity_mapping}
        if unknown_mentions:
            unknown_entity_mention_clusters.append((unknown_mentions, ent))
    return unknown_entity_mention_clusters
