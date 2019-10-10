import datetime
import urllib.parse

TYPE_RESOURCE = 'type_resource'
POSTFIXES = {
    str: 'http://www.w3.org/2001/XMLSchema#string',
    int: 'http://www.w3.org/2001/XMLSchema#integer',
    datetime.datetime: 'http://www.w3.org/2001/XMLSchema#date'
}
RESOURCE_ENCODED_CHARS = ['\\', '\'', '"', '´', '`', '{', '}', '^']
LITERAL_ENCODED_CHARS = ['\\', '\'', '"']


def as_literal_triple(sub: str, pred: str, obj) -> str:
    obj_type = type(obj)
    if obj_type == datetime.datetime:
        obj = obj.strftime('%Y-%m-%d')
    elif obj_type == str:
        obj = _encode_literal_string(obj)
    return _as_triple(sub, pred, obj, obj_type)


def as_object_triple(sub: str, pred: str, obj: str) -> str:
    return _as_triple(sub, pred, obj, TYPE_RESOURCE)


def _as_triple(sub: str, pred: str, obj: str, obj_type) -> str:
    obj_as_string = _resource_to_string(obj) if obj_type == TYPE_RESOURCE else f'"{obj}"^^{_resource_to_string(POSTFIXES[obj_type])}'
    return f'{_resource_to_string(sub)} {_resource_to_string(pred)} {obj_as_string} .\n'


def _resource_to_string(resource: str) -> str:
    for c in RESOURCE_ENCODED_CHARS:
        resource = resource.replace(c, urllib.parse.quote_plus(c))
    return resource if resource.startswith('_:') else f'<{resource}>'


def _encode_literal_string(literal: str) -> str:
    for c in LITERAL_ENCODED_CHARS:
        literal = literal.replace(c, f'\\{c}')
    return literal
