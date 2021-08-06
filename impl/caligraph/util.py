"""Utilities to work with CaLiGraph resources, types, and predicates."""

import utils
import impl.dbpedia.util as dbp_util
import impl.util.rdf as rdf_util
import impl.util.string as str_util
import re


NAMESPACE_CLG_BASE = utils.get_config('caligraph.namespace.base')
NAMESPACE_CLG_ONTOLOGY = utils.get_config('caligraph.namespace.ontology')
NAMESPACE_CLG_RESOURCE = utils.get_config('caligraph.namespace.resource')


def is_clg_type(item: str) -> bool:
    return item.startswith(NAMESPACE_CLG_ONTOLOGY)


def name2clg_type(name: str) -> str:
    return NAMESPACE_CLG_ONTOLOGY + str_util.capitalize(name.strip().replace(' ', '_'))


def name2clg_prop(name: str) -> str:
    return NAMESPACE_CLG_ONTOLOGY + name.strip().replace(' ', '_')


def clg_type2name(clg_type: str) -> str:
    return clg_type[len(NAMESPACE_CLG_ONTOLOGY):].replace('_', ' ')


def dbp_type2clg_type(dbp_type: str) -> str:
    return NAMESPACE_CLG_ONTOLOGY + dbp_type[len(dbp_util.NAMESPACE_DBP_ONTOLOGY):]


def clg_type2dbp_type(clg_type: str) -> str:
    return dbp_util.NAMESPACE_DBP_ONTOLOGY + clg_type[len(NAMESPACE_CLG_ONTOLOGY):]


def is_clg_resource(item: str) -> bool:
    return item.startswith(NAMESPACE_CLG_RESOURCE)


def name2clg_resource(name: str) -> str:
    return rdf_util.name2uri(str(name).strip(), NAMESPACE_CLG_RESOURCE)


def clg_resource2name(clg_resource: str) -> str:
    # deal with weird entities of DBpedia that append '__1', '__2', etc to some entities
    clg_resource = re.sub(r'__\d+$', '', clg_resource)
    name_part = clg_resource[clg_resource.rfind('--')+2:] if '--' in clg_resource else clg_resource[len(NAMESPACE_CLG_RESOURCE):]
    return str_util.regularize_spaces(name_part.replace('_', ' '))


def dbp_resource2clg_resource(dbp_resource: str) -> str:
    return NAMESPACE_CLG_RESOURCE + dbp_resource[len(dbp_util.NAMESPACE_DBP_RESOURCE):]


def clg_resource2dbp_resource(clg_resource: str) -> str:
    return dbp_util.NAMESPACE_DBP_RESOURCE + clg_resource[len(NAMESPACE_CLG_RESOURCE):]
