from typing import List
from enum import Enum
from impl.dbpedia.resource import DbpResourceStore
from collections import Counter
from impl.util.transformer import EntityIndex


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


TYPE_TO_LABEL_MAPPING = {
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


def map_entities_to_pos_labels(entity_chunks: List[List[int]], binary_labels: bool):
    """Transforms the chunks of entity labels into chunks of POS tags."""
    # first find the pos labels for all entities (to avoid duplicate resolution of pos labels)
    all_resource_indices = {idx for chunk in entity_chunks for idx in chunk if idx >= 0}
    pos_tag_by_ent_idx = {idx: _find_pos_tag_for_ent(idx) for idx in all_resource_indices}

    labels = []
    for chunk in entity_chunks:
        # find the majority tag of the chunk which will be the label for all entities in the chunk
        pos_tags = [pos_tag_by_ent_idx[idx] for idx in chunk if idx >= 0]
        most_common_tags = Counter(pos_tags).most_common()
        majority_tag = most_common_tags[0][0] if len(most_common_tags) > 0 else POSLabel.NONE
        entity_value = 1 if binary_labels else majority_tag.value
        labels.append([_map_ent_to_pos_tag(idx, entity_value) for idx in chunk])
    return labels


def _find_pos_tag_for_ent(ent_idx: int) -> POSLabel:
    ent = DbpResourceStore.instance().get_resource_by_idx(ent_idx)
    for t in ent.get_transitive_types():
        if t.name in TYPE_TO_LABEL_MAPPING:
            return TYPE_TO_LABEL_MAPPING[t.name]
    return POSLabel.OTHER


def _map_ent_to_pos_tag(ent_idx: int, entity_value: int):
    match ent_idx:
        case EntityIndex.IGNORE.value:
            return EntityIndex.IGNORE.value
        case EntityIndex.NO_ENTITY.value:
            return POSLabel.NONE.value
        case _:
            return entity_value
