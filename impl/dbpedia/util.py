"""Utilities to work with resources, properties, and classes in DBpedia."""

from impl.util.rdf import Namespace, uri2name


def get_canonical_uri(uri: str) -> str:
    if '__' in uri:
        # some URIs can contain references to specific entities in the page using the postfix '__<some-id-or-number>'
        # we get rid of such specific references as we are only interested in the uri itself
        uri = uri[:uri.find('__')]
    return uri


def is_class(uri: str) -> bool:
    return uri.startswith(Namespace.DBP_ONTOLOGY.value)


def class2name(uri: str) -> str:
    return uri2name(uri, Namespace.DBP_ONTOLOGY.value)


def is_resource(uri: str) -> bool:
    return uri.startswith(Namespace.DBP_RESOURCE.value)


def resource2name(uri: str) -> str:
    return uri2name(uri, Namespace.DBP_RESOURCE.value)


def is_listpage(uri: str) -> bool:
    return uri.startswith(Namespace.DBP_LIST.value)


def is_file(uri: str) -> bool:
    return uri.startswith((Namespace.DBP_FILE.value, Namespace.DBP_IMAGE.value))


def is_category(uri: str) -> bool:
    return uri.startswith(Namespace.DBP_CATEGORY.value)


def category2name(uri: str) -> str:
    return uri2name(uri, Namespace.DBP_CATEGORY.value)


def resource_uri2wikipedia_uri(uri: str) -> str:
    return f'{Namespace.WIKIPEDIA.value}{uri[len(Namespace.DBP_RESOURCE.value):]}' if is_resource(uri) else uri
