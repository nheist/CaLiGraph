from enum import Enum
import impl.dbpedia.util as dbp_util
import impl.dbpedia.store as dbp_store


class POSLabel(Enum):
    NONE = 0
    PERSON = 1
    NORP = 3
    FAC = 5
    ORG = 7
    GPE = 9
    LOC = 11
    PRODUCT = 13
    EVENT = 15
    WORK_OF_ART = 17
    LAW = 19
    LANGUAGE = 21
    SPECIES = 23
    OTHER = 25

    def begin(self):
        return self.value

    def inside(self):
        return self.value + 1 if self.value > 0 else self.value

    @classmethod
    def label_count(cls):
        return len(cls) * 2 - 1


def map_entities_to_pos_labels(entity_chunks: list) -> list:
    """Transforms the chunks of entity labels into chunks of POS tags."""
    return [_map_entity_chunk(chunk) for chunk in entity_chunks]


def _map_entity_chunk(entity_chunk: list) -> list:
    # first find the pos labels for all entities in the chunk (to avoid duplicate resolution of pos labels)
    entity_to_pos_label_mapping = {ent: _find_pos_label_for_ent(ent) for ent in set(entity_chunk)}
    # then map the entities to pos labels
    pos_labels = []
    for idx, ent in enumerate(entity_chunk):
        pos_label = entity_to_pos_label_mapping[ent]
        if idx == 0 or ent != entity_chunk[idx - 1]:
            pos_labels.append(pos_label.begin())
        else:
            pos_labels.append(pos_label.inside())
    return pos_labels


def _find_pos_label_for_ent(ent: str) -> POSLabel:
    if ent is None:
        return POSLabel.NONE
    ent_uri = dbp_util.name2resource(ent)
    ttl_mapping = _get_type_to_label_mapping()
    for t in dbp_store.get_types(ent_uri):
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
