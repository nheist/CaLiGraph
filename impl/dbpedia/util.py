"""Utilities to work with resources, properties, and classes in DBpedia."""

from impl.util.rdf import Namespace, RdfResource, iri2name, name2iri


def get_canonical_uri(uri: str) -> str:
    if '__' in uri:
        # some URIs can contain references to specific entities in the page using the postfix '__<some-id-or-number>'
        # we get rid of such specific references as we are only interested in the uri itself
        uri = uri[:uri.find('__')]
    return uri


def res2dbp_iri(res: RdfResource) -> str:
    return name2iri(res.name, Namespace.DBP_RESOURCE)


def is_class(uri: str) -> bool:
    return uri.startswith(Namespace.DBP_ONTOLOGY.value)


def class2name(uri: str) -> str:
    return iri2name(uri, Namespace.DBP_ONTOLOGY)


def is_resource(uri: str) -> bool:
    return uri.startswith(Namespace.DBP_RESOURCE.value)


def resource2name(uri: str) -> str:
    return iri2name(uri, Namespace.DBP_RESOURCE)


def is_listpage(uri: str) -> bool:
    return uri.startswith(Namespace.DBP_LIST.value)


def is_file(uri: str) -> bool:
    return uri.startswith((Namespace.DBP_FILE.value, Namespace.DBP_IMAGE.value))


def is_category(uri: str) -> bool:
    return uri.startswith(Namespace.DBP_CATEGORY.value)
