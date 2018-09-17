import caligraph.util.rdf as rdf_util

NAMESPACE_DBP_ONTOLOGY = 'http://dbpedia.org/ontology/'
NAMESPACE_DBP_RESOURCE = 'http://dbpedia.org/resource/'


def name2resource(name: str) -> str:
    return rdf_util.name2uri(name, NAMESPACE_DBP_RESOURCE)


def resource2name(resource: str) -> str:
    return rdf_util.uri2name(resource, NAMESPACE_DBP_RESOURCE)


def type2name(dbp_type: str) -> str:
    return rdf_util.uri2name(dbp_type, NAMESPACE_DBP_ONTOLOGY)
