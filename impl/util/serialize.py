"""Serialization of statements into RDF triples."""

from enum import Enum
import datetime
import urllib.parse
import impl.util.string as str_util
import impl.util.rdf as rdf_util
from impl.util.rdf import RdfResource

TYPE_RESOURCE = 'type_resource'
POSTFIXES = {
    int: 'http://www.w3.org/2001/XMLSchema#integer',
    datetime.datetime: 'http://www.w3.org/2001/XMLSchema#date'
}
RESOURCE_ENCODING_EXCEPTIONS = ['#', ':', ',', ';', '(', ')', '\'', '&', '!', '*', '=', '$']
LITERAL_ENCODED_CHARS = ['\\', '"']


def as_literal_triple(sub, pred, obj) -> str:
    """Serialize a triples as literal triple."""
    obj_type = type(obj)
    if obj_type == datetime.datetime:
        obj = obj.strftime('%Y-%m-%d')
    elif obj_type == str:
        obj = _encode_literal_string(obj)
    return _as_triple(sub, pred, obj, obj_type)


def as_object_triple(sub, pred, obj) -> str:
    """Serialize a triples as object triple."""
    return _as_triple(sub, pred, obj, TYPE_RESOURCE)


def _as_triple(sub, pred, obj, obj_type) -> str:
    sub_str = _resource_to_string(sub)
    pred_str = _resource_to_string(pred)
    if obj_type == TYPE_RESOURCE:
        obj_str = _resource_to_string(obj)
    else:
        obj_str = f'"{obj}"'
        if obj_type in POSTFIXES:
            obj_str += f'^^{_resource_to_string(POSTFIXES[obj_type])}'
    return f'{sub_str} {pred_str} {obj_str} .\n'


def _resource_to_string(resource) -> str:
    if isinstance(resource, Enum):
        resource = resource.value
    if isinstance(resource, str):
        if '/wiki/' in resource:
            prefix = resource[:resource.rfind('/wiki/') + len('/wiki/')]
        else:
            prefix = resource[:resource.rfind('/') + 1]
        res_name = resource[len(prefix):]
    elif isinstance(resource, RdfResource):
        prefix = resource.get_namespace()
        res_name = resource.name
    else:
        raise ValueError(f'Can not convert a resource of type {type(resource)} to string.')
    return f'<{rdf_util.name2iri(_encode_resource(res_name), prefix)}>'


def _encode_resource(resource) -> str:
    try:
        res_name = urllib.parse.quote(str(resource))
    except TypeError as e:
        raise e
    for char in RESOURCE_ENCODING_EXCEPTIONS:
        res_name = res_name.replace(urllib.parse.quote(char), char)
    return res_name


def _encode_literal_string(literal: str) -> str:
    for c in LITERAL_ENCODED_CHARS:
        literal = literal.replace(c, f'\\{c}')
    return str_util.regularize_spaces(literal)
