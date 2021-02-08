"""Utilities to work with resources, properties, and types in DBpedia."""

import impl.util.rdf as rdf_util

NAMESPACE_DBP_ONTOLOGY = 'http://dbpedia.org/ontology/'
NAMESPACE_DBP_RESOURCE = 'http://dbpedia.org/resource/'


NER_LABEL_MAPPING = {
    'PERSON': ['http://dbpedia.org/ontology/Person', 'http://dbpedia.org/ontology/Deity'],
    'NORP': ['http://dbpedia.org/ontology/PoliticalParty', 'http://dbpedia.org/ontology/Family', 'http://dbpedia.org/ontology/EthnicGroup'],
    'FAC': ['http://dbpedia.org/ontology/ArchitecturalStructure', 'http://dbpedia.org/ontology/Mine', 'http://dbpedia.org/ontology/Monument'],
    'ORG': ['http://dbpedia.org/ontology/Organisation'],
    'GPE': ['http://dbpedia.org/ontology/PopulatedPlace'],
    'LOC': ['http://dbpedia.org/ontology/Place', 'http://dbpedia.org/ontology/Location'],
    'PRODUCT': ['http://dbpedia.org/ontology/Food', 'http://dbpedia.org/ontology/MeanOfTransportation', 'http://dbpedia.org/ontology/Software', 'http://dbpedia.org/ontology/Device'],
    'EVENT': ['http://dbpedia.org/ontology/Event'],
    'WORK_OF_ART': ['http://dbpedia.org/ontology/Work', 'http://dbpedia.org/ontology/Award'],
    'LAW': ['http://dbpedia.org/ontology/Law', 'http://dbpedia.org/ontology/LegalCase', 'http://dbpedia.org/ontology/Treaty'],
    'LANGUAGE': ['http://dbpedia.org/ontology/Language'],
    'SPECIES': ['http://dbpedia.org/ontology/Species']
}


def object2name(dbp_object: str) -> str:
    if dbp_object.startswith(NAMESPACE_DBP_ONTOLOGY):
        return type2name(dbp_object)
    else:
        return resource2name(dbp_object)


def name2resource(name: str) -> str:
    return rdf_util.name2uri(name, NAMESPACE_DBP_RESOURCE)


def resource2name(resource: str) -> str:
    return rdf_util.uri2name(resource, NAMESPACE_DBP_RESOURCE)


def name2type(name: str) -> str:
    return rdf_util.name2uri(name, NAMESPACE_DBP_ONTOLOGY)


def type2name(dbp_type: str) -> str:
    return rdf_util.uri2name(dbp_type, NAMESPACE_DBP_ONTOLOGY)


def is_dbp_type(dbp_type: str) -> bool:
    return dbp_type.startswith(NAMESPACE_DBP_ONTOLOGY)


def is_dbp_resource(dbp_resource: str) -> bool:
    return dbp_resource.startswith(NAMESPACE_DBP_RESOURCE)


def is_file_resource(dbp_object: str) -> bool:
    return dbp_object.startswith(('File:', 'Image:'), len(NAMESPACE_DBP_RESOURCE))


def dbp_resource2wikipedia_resource(dbp_resource: str) -> str:
    if not is_dbp_resource(dbp_resource):
        return dbp_resource
    return f'http://en.wikipedia.org/wiki/{dbp_resource[len(NAMESPACE_DBP_RESOURCE):]}'
