from typing import List
from enum import Enum
from impl.dbpedia.resource import DbpEntity, DbpResourceStore
from collections import Counter
from .word_tokenize import WordTokenizerSpecialLabel


class POSLabel(Enum):
    NONE = 0
    PERSON = 1
    NORP = 2
    FAC = 3
    ORG = 4
    GPE = 5
    LOC = 6
    PRODUCT = 7
    EVENT = 8
    WORK_OF_ART = 9
    LAW = 10
    LANGUAGE = 11
    SPECIES = 12
    OTHER = 13


def map_entities_to_pos_labels(entity_chunks: List[List[int]], single_tag_prediction=False):
    """Transforms the chunks of entity labels into chunks of POS tags."""
    # first find the pos labels for all entities (to avoid duplicate resolution of pos labels)
    all_ents = {DbpResourceStore.instance().get_resource_by_idx(idx) for chunk in entity_chunks for idx in chunk if idx >= 0}
    pos_tags = [_find_pos_tag_for_ent(ent) for ent in all_ents if isinstance(ent, DbpEntity)]
    # then find the majority tag of the chunk which will be the label for all entities in the chunk
    most_common_tags = Counter(pos_tags).most_common()
    majority_tag = most_common_tags[0][0] if len(most_common_tags) > 0 else POSLabel.NONE
    entity_value = 1 if single_tag_prediction else majority_tag.value

    labels = []
    for chunk in entity_chunks:
        chunk_labels = [_map_label_to_pos_tag(label, entity_value) for label in chunk]
        labels.append((chunk_labels, majority_tag.value) if single_tag_prediction else chunk_labels)
    return labels


def _find_pos_tag_for_ent(ent: DbpEntity) -> POSLabel:
    ttl_mapping = _get_type_to_label_mapping()
    for t in ent.get_transitive_types():
        if t in ttl_mapping:
            return ttl_mapping[t]
    return POSLabel.OTHER


def _get_type_to_label_mapping() -> dict:
    return {
        # PERSON
        'http://dbpedia.org/ontology/Person': POSLabel.PERSON,
        'http://dbpedia.org/ontology/Deity': POSLabel.PERSON,
        # NORP
        'http://dbpedia.org/ontology/PoliticalParty': POSLabel.NORP,
        'http://dbpedia.org/ontology/Family': POSLabel.NORP,
        'http://dbpedia.org/ontology/EthnicGroup': POSLabel.NORP,
        # FAC
        'http://dbpedia.org/ontology/ArchitecturalStructure': POSLabel.FAC,
        'http://dbpedia.org/ontology/Mine': POSLabel.FAC,
        'http://dbpedia.org/ontology/Monument': POSLabel.FAC,
        # ORG
        'http://dbpedia.org/ontology/Organisation': POSLabel.ORG,
        # GPE
        'http://dbpedia.org/ontology/PopulatedPlace': POSLabel.GPE,
        # LOC
        'http://dbpedia.org/ontology/Place': POSLabel.LOC,
        'http://dbpedia.org/ontology/Location': POSLabel.LOC,
        # PRODUCT
        'http://dbpedia.org/ontology/Food': POSLabel.PRODUCT,
        'http://dbpedia.org/ontology/MeanOfTransportation': POSLabel.PRODUCT,
        'http://dbpedia.org/ontology/Software': POSLabel.PRODUCT,
        'http://dbpedia.org/ontology/Device': POSLabel.PRODUCT,
        # EVENT
        'http://dbpedia.org/ontology/Event': POSLabel.EVENT,
        # WORK_OF_ART
        'http://dbpedia.org/ontology/Work': POSLabel.WORK_OF_ART,
        'http://dbpedia.org/ontology/Award': POSLabel.WORK_OF_ART,
        # LAW
        'http://dbpedia.org/ontology/Law': POSLabel.LAW,
        'http://dbpedia.org/ontology/LegalCase': POSLabel.LAW,
        'http://dbpedia.org/ontology/Treaty': POSLabel.LAW,
        # LANGUAGE
        'http://dbpedia.org/ontology/Language': POSLabel.LANGUAGE,
        # SPECIES
        'http://dbpedia.org/ontology/Species': POSLabel.SPECIES
    }


def _map_label_to_pos_tag(label: int, entity_value: int):
    match label:
        case WordTokenizerSpecialLabel.IGNORE.value:
            return WordTokenizerSpecialLabel.IGNORE.value
        case WordTokenizerSpecialLabel.NO_ENTITY.value:
            return POSLabel.NONE.value
        case _:
            return entity_value
