import impl.util.rdf as rdf_util

NAMESPACE_DBP_ONTOLOGY = 'http://dbpedia.org/ontology/'
NAMESPACE_DBP_RESOURCE = 'http://dbpedia.org/resource/'


def object2name(dbp_object: str) -> str:
    if dbp_object.startswith(NAMESPACE_DBP_ONTOLOGY):
        return type2name(dbp_object)
    else:
        return resource2name(dbp_object)


def name2resource(name: str) -> str:
    return rdf_util.name2uri(name, NAMESPACE_DBP_RESOURCE)


def resource2name(resource: str) -> str:
    return rdf_util.uri2name(resource, NAMESPACE_DBP_RESOURCE)


def type2name(dbp_type: str) -> str:
    return rdf_util.uri2name(dbp_type, NAMESPACE_DBP_ONTOLOGY)


def is_dbp_type(dbp_type: str) -> bool:
    return dbp_type.startswith(NAMESPACE_DBP_ONTOLOGY)


def is_dbp_resource(dbp_resource: str) -> bool:
    return dbp_resource.startswith(NAMESPACE_DBP_RESOURCE)
