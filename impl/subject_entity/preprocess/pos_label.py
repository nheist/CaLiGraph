from enum import Enum
import impl.dbpedia.util as dbp_util
import impl.dbpedia.store as dbp_store
from collections import Counter


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


def map_entities_to_pos_labels(entity_chunks: list, single_tag_prediction=False) -> list:
    """Transforms the chunks of entity labels into chunks of POS tags."""
    # first find the pos labels for all entities (to avoid duplicate resolution of pos labels)
    all_ents = {e for chunk in entity_chunks for e in chunk}
    entity_to_pos_tag_mapping = {ent: _find_pos_tag_for_ent(ent) for ent in all_ents if ent is not None and type(ent) != int}
    # then find the majority tag of the chunk which will be the label for all entities in the chunk
    most_common_tags = Counter([v for v in entity_to_pos_tag_mapping.values()]).most_common()
    majority_tag = most_common_tags[0][0] if len(most_common_tags) > 0 else POSLabel.NONE
    entity_value = 1 if single_tag_prediction else majority_tag.value

    labels = []
    for chunk in entity_chunks:
        chunk_labels = []
        for idx, ent in enumerate(chunk):
            if type(ent) == int:
                chunk_labels.append(ent)
            elif ent is None:
                chunk_labels.append(POSLabel.NONE.value)
            else:
                chunk_labels.append(entity_value)
        labels.append((chunk_labels, majority_tag.value) if single_tag_prediction else chunk_labels)
    return labels


def _find_pos_tag_for_ent(ent) -> POSLabel:
    ent_uri = dbp_util.name2resource(ent)
    ttl_mapping = _get_type_to_label_mapping()
    for t in dbp_store.get_transitive_types(ent_uri):
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
