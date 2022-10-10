from typing import List
from collections import Counter
from impl.util.nlp import EntityTypeLabel
from impl.util.transformer import EntityIndex
from impl.dbpedia.resource import DbpResourceStore


def map_entities_to_type_labels(entity_chunks: List[List[int]], binary_labels: bool):
    """Transforms the chunks of entity labels into chunks of POS tags."""
    labels = []
    dbr = DbpResourceStore.instance()
    for chunk in entity_chunks:
        # find the majority tag of the chunk which will be the label for all entities in the chunk
        entity_types = [dbr.get_type_label(idx) for idx in chunk if idx >= 0]
        most_common_types = Counter(entity_types).most_common()
        majority_type = most_common_types[0][0] if len(most_common_types) > 0 else EntityTypeLabel.NONE
        entity_value = 1 if binary_labels else majority_type.value
        labels.append([_map_ent_to_type_label(idx, entity_value) for idx in chunk])
    return labels


def _map_ent_to_type_label(ent_idx: int, entity_value: int):
    if ent_idx == EntityIndex.IGNORE.value:
        return EntityIndex.IGNORE.value
    elif ent_idx == EntityIndex.NO_ENTITY.value:
        return EntityTypeLabel.NONE.value
    return entity_value
