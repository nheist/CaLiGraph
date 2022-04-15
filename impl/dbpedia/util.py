"""Utilities to work with resources, properties, and classes in DBpedia."""

from impl.util.rdf import Namespace, RdfResource, iri2name, name2iri


def get_canonical_iri(iri: str) -> str:
    # some URIs can contain references to specific entities in the page using the postfix '__<some-id-or-number>'
    # we get rid of such specific references as we are only interested in the uri itself
    return iri[:iri.find('__')] if '__' in iri else iri


def res2dbp_iri(res: RdfResource) -> str:
    return name2iri(res.name, Namespace.DBP_RESOURCE)


def is_class_iri(iri: str) -> bool:
    return iri.startswith(Namespace.DBP_ONTOLOGY.value)


def class_iri2name(iri: str) -> str:
    return iri2name(iri, Namespace.DBP_ONTOLOGY)


def is_resource_iri(iri: str) -> bool:
    return iri.startswith(Namespace.DBP_RESOURCE.value)


def resource_iri2name(iri: str) -> str:
    return iri2name(iri, Namespace.DBP_RESOURCE)


def name2resource_iri(name: str) -> str:
    return name2iri(name, Namespace.DBP_RESOURCE)


def is_listpage_iri(iri: str) -> bool:
    return iri.startswith(Namespace.DBP_LIST.value)


def is_file_iri(iri: str) -> bool:
    return iri.startswith((Namespace.DBP_FILE.value, Namespace.DBP_IMAGE.value))


def is_category_iri(iri: str) -> bool:
    return iri.startswith(Namespace.DBP_CATEGORY.value)


def is_entity_name(name: str) -> bool:
    invalid_prefixes = (Namespace.PREFIX_LIST.value, Namespace.PREFIX_FILE.value, Namespace.PREFIX_IMAGE.value,
                        Namespace.PREFIX_CATEGORY.value, Namespace.PREFIX_TEMPLATE.value)
    return not name.startswith(invalid_prefixes)
