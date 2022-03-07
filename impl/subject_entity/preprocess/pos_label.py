from enum import Enum
import impl.dbpedia.util as dbp_util
import impl.dbpedia.store as dbp_store
from collections import Counter


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


def map_entities_to_pos_labels(entity_chunks: list, use_majority_tag=False) -> list:
    """Transforms the chunks of entity labels into chunks of POS tags."""
    # first find the pos labels for all entities (to avoid duplicate resolution of pos labels)
    all_ents = {e for chunk in entity_chunks for e in chunk}
    entity_to_pos_tag_mapping = {ent: _find_pos_tag_for_ent(ent) for ent in all_ents if type(ent) != int}
    majority_tag = Counter([v for v in entity_to_pos_tag_mapping.values() if v != POSLabel.NONE]).most_common(1)[0][0]

    pos_labels = []
    for chunk in entity_chunks:
        chunk_labels = []
        for idx, ent in enumerate(chunk):
            if type(ent) == int:
                chunk_labels.append(ent)
                continue
            pos_tag = majority_tag if use_majority_tag else entity_to_pos_tag_mapping[ent]
            if idx == 0 or ent != chunk[idx - 1]:
                chunk_labels.append(pos_tag.begin())
            else:
                chunk_labels.append(pos_tag.inside())
        pos_labels.append(chunk_labels)
    return pos_labels


def _find_pos_tag_for_ent(ent) -> POSLabel:
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
