from collections import namedtuple
import bz2
import re
from typing import Iterator
from collections import defaultdict
import functools

# predicates
PREDICATE_EQUIVALENT_CLASS = 'http://www.w3.org/2002/07/owl#equivalentClass'
PREDICATE_WAS_DERIVED_FROM = 'https://www.w3.org/ns/prov#wasDerivedFrom'
PREDICATE_BROADER = 'http://www.w3.org/2004/02/skos/core#broader'
PREDICATE_TYPE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
PREDICATE_LABEL = 'http://www.w3.org/2004/02/skos/core#prefLabel'
PREDICATE_SUBJECT = 'http://purl.org/dc/terms/subject'
PREDICATE_SUBCLASS_OF = 'http://www.w3.org/2000/01/rdf-schema#subClassOf'
PREDICATE_DISJOINT_WITH = 'http://www.w3.org/2002/07/owl#disjointWith'
PREDICATE_REDIRECTS = 'http://dbpedia.org/ontology/wikiPageRedirects'
PREDICATE_DOMAIN = 'http://www.w3.org/2000/01/rdf-schema#domain'
PREDICATE_SAME_AS = 'http://www.w3.org/2002/07/owl#sameAs'
PREDICATE_EQUIVALENT_PROPERTY = 'http://www.w3.org/2002/07/owl#equivalentProperty'

# classes
CLASS_OWL_THING = 'http://www.w3.org/2002/07/owl#Thing'

# auxiliary structures
Triple = namedtuple('Triple', 'sub pred obj')


def uri2name(uri: str, prefix: str) -> str:
    return uri[len(prefix):].replace('_', ' ')


def name2uri(name: str, prefix: str) -> str:
    return prefix + name.replace(' ', '_')


def get_literal_value(literal_string: str) -> str:
    string_start = literal_string.find('"')+1
    string_end = literal_string.rfind('"')
    return literal_string[string_start:string_end]


def format_object_triple(sub, pred, obj):
    return '<{}> <{}> <{}> .\n'.format(sub, pred, obj)


def parse_triples_from_file(filepath: str) -> Iterator[Triple]:
    object_pattern = re.compile(b'\<(.*)\> \<(.*)\> \<(.*)\> \.\\n')
    literal_pattern = re.compile(b'\<(.*)\> \<(.*)\> "(.*)"(?:\^\^.*|@en.*)? \.\\n')

    open_file = bz2.open if filepath.endswith('bz2') else open
    with open_file(filepath, mode="rb") as file_reader:
        for line in file_reader:
            object_triple = object_pattern.match(line)
            if object_triple:
                [sub, pred, obj] = object_triple.groups()
                yield Triple(sub=sub.decode('utf-8'), pred=pred.decode('utf-8'), obj=obj.decode('utf-8'))
            else:
                literal_triple = literal_pattern.match(line)
                if literal_triple:
                    [sub, pred, obj] = literal_triple.groups()
                    yield Triple(sub=sub.decode('utf-8'), pred=pred.decode('utf-8'), obj=obj.decode('utf-8'))


def create_multi_val_dict_from_rdf(filepaths: list, predicate: str, reverse_key=False, reflexive=False) -> dict:
    data_dict = defaultdict(set)
    for fp in filepaths:
        for triple in parse_triples_from_file(fp):
            if triple.pred == predicate:
                if reflexive or reverse_key:
                    data_dict[triple.obj].add(triple.sub)
                if reflexive or not reverse_key:
                    data_dict[triple.sub].add(triple.obj)
    return data_dict


def create_single_val_dict_from_rdf(filepaths: list, predicate: str, reverse_key=False, reflexive=False) -> dict:
    data_dict = {}
    for fp in filepaths:
        for triple in parse_triples_from_file(fp):
            if triple.pred == predicate:
                if reflexive or reverse_key:
                    data_dict[triple.obj] = triple.sub
                elif reflexive or not reverse_key:
                    data_dict[triple.sub] = triple.obj
    return data_dict


def create_dict_from_rdf(filepaths: list, valid_predicates: set = None) -> dict:
    data_dict = defaultdict(functools.partial(defaultdict, set))
    for fp in filepaths:
        for triple in parse_triples_from_file(fp):
            if not valid_predicates or triple.pred in valid_predicates:
                data_dict[triple.sub][triple.pred].add(triple.obj)
    return data_dict


def create_count_dict(iterables) -> dict:
    count_dict = defaultdict(int)
    for i in iterables:
        for entry in i:
            count_dict[entry] += 1
    return count_dict
