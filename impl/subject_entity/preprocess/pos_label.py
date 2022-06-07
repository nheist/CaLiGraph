from typing import List
from enum import Enum
from impl.dbpedia.resource import DbpEntity, DbpResourceStore
from collections import Counter
from .word_tokenize import WordTokenizedSpecialLabel


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
    dbr = DbpResourceStore.instance()
    all_resources = [dbr.get_resource_by_idx(idx) for chunk in entity_chunks for idx in chunk if idx >= 0]
    pos_tags_by_ent = {res: _find_pos_tag_for_ent(res) for res in set(all_resources) if isinstance(res, DbpEntity)}
    pos_tags = [pos_tags_by_ent[res] for res in all_resources if res in pos_tags_by_ent]
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
        if t.name in ttl_mapping:
            return ttl_mapping[t.name]
    return POSLabel.OTHER


def _get_type_to_label_mapping() -> dict:
    return {
        # PERSON
        'Person': POSLabel.PERSON,
        'Deity': POSLabel.PERSON,
        # NORP
        'PoliticalParty': POSLabel.NORP,
        'Family': POSLabel.NORP,
        'EthnicGroup': POSLabel.NORP,
        # FAC
        'ArchitecturalStructure': POSLabel.FAC,
        'Mine': POSLabel.FAC,
        'Monument': POSLabel.FAC,
        # ORG
        'Organisation': POSLabel.ORG,
        # GPE
        'PopulatedPlace': POSLabel.GPE,
        # LOC
        'Place': POSLabel.LOC,
        'Location': POSLabel.LOC,
        # PRODUCT
        'Food': POSLabel.PRODUCT,
        'MeanOfTransportation': POSLabel.PRODUCT,
        'Software': POSLabel.PRODUCT,
        'Device': POSLabel.PRODUCT,
        # EVENT
        'Event': POSLabel.EVENT,
        # WORK_OF_ART
        'Work': POSLabel.WORK_OF_ART,
        'Award': POSLabel.WORK_OF_ART,
        # LAW
        'Law': POSLabel.LAW,
        'LegalCase': POSLabel.LAW,
        'Treaty': POSLabel.LAW,
        # LANGUAGE
        'Language': POSLabel.LANGUAGE,
        # SPECIES
        'Species': POSLabel.SPECIES
    }


def _map_label_to_pos_tag(label: int, entity_value: int):
    match label:
        case WordTokenizedSpecialLabel.IGNORE.value:
            return WordTokenizedSpecialLabel.IGNORE.value
        case WordTokenizedSpecialLabel.NO_ENTITY.value:
            return POSLabel.NONE.value
        case _:
            return entity_value
