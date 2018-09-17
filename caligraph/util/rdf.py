from collections import namedtuple
import bz2
import re
import util
from collections import defaultdict

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

# classes
CLASS_OWL_THING = 'http://www.w3.org/2002/07/owl#Thing'

# auxiliary structures
Triple = namedtuple('Triple', 'sub pred obj')
DirectedEdge = namedtuple('DirectedEdge', 'parent child')


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


def parse_triples_from_file(filepath: str) -> list:
    open_file = bz2.open if filepath.endswith('bz2') else open
    with open_file(filepath, mode="rt", encoding="utf-8") as file_reader:
        triple_lines = file_reader.readlines()

    triples = []
    object_pattern = re.compile('\<(.*)\> \<(.*)\> \<(.*)\> \.\\n')
    literal_pattern = re.compile('\<(.*)\> \<(.*)\> (.*) \.\\n')

    for line in triple_lines:
        object_triple = object_pattern.match(line)
        if object_triple is not None:
            [sub, pred, obj] = object_triple.groups()
            triples.append(Triple(sub=sub, pred=pred, obj=obj))
        else:
            literal_triple = literal_pattern.match(line)
            if literal_triple is not None:
                [sub, pred, obj] = literal_triple.groups()
                triples.append(Triple(sub=sub, pred=pred, obj=obj))
            else:
                util.get_logger().debug('rdfutil: could not parse line: {}'.format(line))

    return triples


def create_multi_val_dict_from_rdf(filepaths: list, predicate: str, reverse_key=False, reflexive=False) -> dict:
    data_dict = defaultdict(set)
    for fp in filepaths:
        for triple in parse_triples_from_file(fp):
            if triple.pred == predicate:
                if reflexive or reverse_key:
                    data_dict[triple.obj].add(triple.sub)
                elif reflexive or not reverse_key:
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
