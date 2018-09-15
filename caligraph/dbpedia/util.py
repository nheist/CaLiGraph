# namespaces
NAMESPACE_DBP_ONTOLOGY = 'http://dbpedia.org/ontology/'
NAMESPACE_DBP_RESOURCE = 'http://dbpedia.org/resource/'
NAMESPACE_DBP_CATEGORY = NAMESPACE_DBP_RESOURCE + 'Category:'


def label2resource(label: str) -> str:
    return NAMESPACE_DBP_RESOURCE + label.replace(' ', '_')


def resource2label(resource: str) -> str:
    return resource[len(NAMESPACE_DBP_RESOURCE):].replace('_', ' ')


def type2label(dbp_type: str) -> str:
    return dbp_type[len(NAMESPACE_DBP_ONTOLOGY):]
