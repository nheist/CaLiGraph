from typing import List
from collections import Counter
from impl.util.nlp import EntityTypeLabel
from impl.util.transformer import EntityIndex
from impl.dbpedia.resource import DbpResourceStore


TYPE_TO_LABEL_MAPPING = {
    # PERSON
    'Person': EntityTypeLabel.PERSON,
    'Deity': EntityTypeLabel.PERSON,
    # NORP
    'PoliticalParty': EntityTypeLabel.NORP,
    'Family': EntityTypeLabel.NORP,
    'EthnicGroup': EntityTypeLabel.NORP,
    # FAC
    'ArchitecturalStructure': EntityTypeLabel.FAC,
    'Mine': EntityTypeLabel.FAC,
    'Monument': EntityTypeLabel.FAC,
    # ORG
    'Organisation': EntityTypeLabel.ORG,
    # GPE
    'PopulatedPlace': EntityTypeLabel.GPE,
    # LOC
    'Place': EntityTypeLabel.LOC,
    'Location': EntityTypeLabel.LOC,
    # PRODUCT
    'Food': EntityTypeLabel.PRODUCT,
    'MeanOfTransportation': EntityTypeLabel.PRODUCT,
    'Software': EntityTypeLabel.PRODUCT,
    'Device': EntityTypeLabel.PRODUCT,
    # EVENT
    'Event': EntityTypeLabel.EVENT,
    # WORK_OF_ART
    'Work': EntityTypeLabel.WORK_OF_ART,
    'Award': EntityTypeLabel.WORK_OF_ART,
    # LAW
    'Law': EntityTypeLabel.LAW,
    'LegalCase': EntityTypeLabel.LAW,
    'Treaty': EntityTypeLabel.LAW,
    # LANGUAGE
    'Language': EntityTypeLabel.LANGUAGE,
    # SPECIES
    'Species': EntityTypeLabel.SPECIES
}


def map_entities_to_type_labels(entity_chunks: List[List[int]], binary_labels: bool):
    """Transforms the chunks of entity labels into chunks of POS tags."""
    # first find the pos labels for all entities (to avoid duplicate resolution of pos labels)
    all_resource_indices = {idx for chunk in entity_chunks for idx in chunk if idx >= 0}
    types_by_ent_idx = {idx: _find_type_for_ent(idx) for idx in all_resource_indices}

    labels = []
    for chunk in entity_chunks:
        # find the majority tag of the chunk which will be the label for all entities in the chunk
        entity_types = [types_by_ent_idx[idx] for idx in chunk if idx >= 0]
        most_common_types = Counter(entity_types).most_common()
        majority_type = most_common_types[0][0] if len(most_common_types) > 0 else EntityTypeLabel.NONE
        entity_value = 1 if binary_labels else majority_type.value
        labels.append([_map_ent_to_type_label(idx, entity_value) for idx in chunk])
    return labels


def _find_type_for_ent(ent_idx: int) -> EntityTypeLabel:
    ent = DbpResourceStore.instance().get_resource_by_idx(ent_idx)
    for t in ent.get_transitive_types():
        if t.name in TYPE_TO_LABEL_MAPPING:
            return TYPE_TO_LABEL_MAPPING[t.name]
    return EntityTypeLabel.OTHER


def _map_ent_to_type_label(ent_idx: int, entity_value: int):
    match ent_idx:
        case EntityIndex.IGNORE.value:
            return EntityIndex.IGNORE.value
        case EntityIndex.NO_ENTITY.value:
            return EntityTypeLabel.NONE.value
        case _:
            return entity_value
