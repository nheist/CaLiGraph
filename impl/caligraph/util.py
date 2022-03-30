"""Utilities to work with CaLiGraph resources, types, and predicates."""

from typing import Optional
from impl.util.rdf import Namespace
from impl.dbpedia.resource import DbpResource, DbpResourceStore
from impl.dbpedia.ontology import DbpClass, DbpOntologyStore
import impl.util.rdf as rdf_util
import impl.util.string as str_util


def is_clg_type(uri: str) -> bool:
    return uri.startswith(Namespace.CLG_ONTOLOGY.value)


def name2clg_type(name: str) -> str:
    return Namespace.CLG_ONTOLOGY.value + str_util.capitalize(name.strip().replace(' ', '_'))


def name2clg_prop(name: str) -> str:
    return Namespace.CLG_ONTOLOGY.value + name.strip().replace(' ', '_')


def clg_class2name(clg_class: str) -> str:
    return rdf_util.uri2name(clg_class, Namespace.CLG_ONTOLOGY.value)


def dbp_class2clg_class(dbp_class: DbpClass) -> str:
    return Namespace.CLG_ONTOLOGY.value + dbp_class.name


def clg_class2dbp_class(clg_class: str) -> DbpClass:
    return DbpOntologyStore.instance().get_class_by_name(clg_class2name(clg_class))


def is_clg_resource(uri: str) -> bool:
    return uri.startswith(Namespace.CLG_RESOURCE.value)


def name2clg_resource(name: str) -> str:
    return rdf_util.name2uri(str(name).strip(), Namespace.CLG_RESOURCE.value)


def clg_resource2name(clg_resource: str) -> str:
    name_part = clg_resource[clg_resource.rfind('--')+2:] if '--' in clg_resource else clg_resource[len(Namespace.CLG_RESOURCE.value):]
    return str_util.regularize_spaces(name_part.replace('_', ' '))


def dbp_resource2clg_resource(dbp_resource: DbpResource) -> str:
    return Namespace.CLG_RESOURCE.value + dbp_resource.name


def clg_resource2dbp_resource(clg_resource: str) -> Optional[DbpResource]:
    dbr = DbpResourceStore.instance()
    res_name = clg_resource2name(clg_resource)
    return dbr.get_resource_by_name(res_name) if dbr.has_resource_with_name(res_name) else None


def clg_resource2dbp_iri(clg_resource: str) -> str:
    return f'{Namespace.DBP_RESOURCE.value}{clg_resource[len(Namespace.CLG_RESOURCE.value):]}'
