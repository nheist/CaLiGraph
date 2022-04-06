"""Functionality for handling RDF (parsing, conversion, formatting)."""

from typing import Iterator, Optional, Union
from collections import namedtuple
import bz2
import re
from collections import defaultdict, Counter
import functools
import urllib.parse
import impl.util.string as str_util
from enum import Enum
from dataclasses import dataclass

import utils


class Namespace(Enum):
    OWL = 'http://www.w3.org/2002/07/owl#'
    WIKIPEDIA = 'http://en.wikipedia.org/wiki/'

    PREFIX_TEMPLATE = 'Template:'
    PREFIX_CATEGORY = 'Category:'
    PREFIX_FILE = 'File:'
    PREFIX_IMAGE = 'Image:'
    PREFIX_LIST = 'List_of_'
    PREFIX_LISTS = 'Lists_of_'

    DBP_ONTOLOGY = 'http://dbpedia.org/ontology/'
    DBP_RESOURCE = 'http://dbpedia.org/resource/'
    DBP_TEMPLATE = DBP_RESOURCE + PREFIX_TEMPLATE
    DBP_CATEGORY = DBP_RESOURCE + PREFIX_CATEGORY
    DBP_FILE = DBP_RESOURCE + PREFIX_FILE
    DBP_IMAGE = DBP_RESOURCE + PREFIX_IMAGE
    DBP_LIST = DBP_RESOURCE + PREFIX_LIST

    CLG_ONTOLOGY = 'http://caligraph.org/ontology/'
    CLG_RESOURCE = 'http://caligraph.org/resource/'


class RdfClass(Enum):
    OWL_THING = 'http://www.w3.org/2002/07/owl#Thing'
    OWL_CLASS = 'http://www.w3.org/2002/07/owl#Class'
    OWL_NAMED_INDIVIDUAL = 'http://www.w3.org/2002/07/owl#NamedIndividual'
    OWL_OBJECT_PROPERTY = 'http://www.w3.org/2002/07/owl#ObjectProperty'
    OWL_DATATYPE_PROPERTY = 'http://www.w3.org/2002/07/owl#DatatypeProperty'


class RdfPredicate(Enum):
    EQUIVALENT_CLASS = 'http://www.w3.org/2002/07/owl#equivalentClass'
    WAS_DERIVED_FROM = 'http://www.w3.org/ns/prov#wasDerivedFrom'
    BROADER = 'http://www.w3.org/2004/02/skos/core#broader'
    PREFLABEL = 'http://www.w3.org/2004/02/skos/core#prefLabel'
    ALTLABEL = 'http://www.w3.org/2004/02/skos/core#altLabel'
    LABEL = 'http://www.w3.org/2000/01/rdf-schema#label'
    COMMENT = 'http://www.w3.org/2000/01/rdf-schema#comment'
    TYPE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    SUBJECT = 'http://purl.org/dc/terms/subject'
    SUBCLASS_OF = 'http://www.w3.org/2000/01/rdf-schema#subClassOf'
    DISJOINT_WITH = 'http://www.w3.org/2002/07/owl#disjointWith'
    REDIRECTS = 'http://dbpedia.org/ontology/wikiPageRedirects'
    DISAMBIGUATES = 'http://dbpedia.org/ontology/wikiPageDisambiguates'
    DOMAIN = 'http://www.w3.org/2000/01/rdf-schema#domain'
    RANGE = 'http://www.w3.org/2000/01/rdf-schema#range'
    SAME_AS = 'http://www.w3.org/2002/07/owl#sameAs'
    EQUIVALENT_PROPERTY = 'http://www.w3.org/2002/07/owl#equivalentProperty'
    WIKILINK = 'http://dbpedia.org/ontology/wikiPageWikiLink'
    ANCHOR_TEXT = 'http://dbpedia.org/ontology/wikiPageWikiLinkText'


@dataclass(frozen=True)
class RdfResource:
    idx: int
    name: str
    is_meta: bool  # a resource is a meta resource if it is a redirect or a disambiguation

    def __lt__(self, other):
        return self.idx < other.idx

    def __hash__(self):
        return id(self)  # use object id as hash is okay here as we are centrally managing the RdfResources

    def get_label(self) -> str:
        label = self._get_store().get_label(self) or self.name
        prefix = self._get_prefix()
        if prefix and label.startswith(prefix):
            label = label[len(prefix):]
        return label

    @classmethod
    def get_namespace(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def _get_store(cls):
        raise NotImplementedError()

    @classmethod
    def _get_prefix(cls) -> str:
        return ''


# auxiliary structures
Triple = namedtuple('Triple', 'sub pred obj is_literal')


def res2iri(res: RdfResource) -> str:
    return name2iri(res.name, res.get_namespace())


def res2wiki_iri(res: RdfResource) -> str:
    return name2iri(res.name, Namespace.WIKIPEDIA)


def iri2name(iri: str, prefix: Union[str, Enum]) -> str:
    if iri == RdfClass.OWL_THING.value:
        return 'Thing'
    prefix = prefix.value if isinstance(prefix, Enum) else prefix
    return iri[len(str(prefix)):]


def name2iri(name: str, prefix: Union[str, Enum]) -> str:
    if name == 'Thing':
        return RdfClass.OWL_THING.value
    prefix = prefix.value if isinstance(prefix, Enum) else prefix
    return prefix + name


def parse_triples_from_file(filepath: str) -> Iterator[Triple]:
    """Parse triples from file using a regular expression that is only guaranteed to work for DBpedia files."""
    object_pattern = re.compile(rb'<(.+)> <(.+)> <(.+)> \.\s*\n')
    literal_pattern = re.compile(rb'<(.+)> <(.+)> "(.+)"(?:\^\^.*|@en.*)? \.\s*\n')

    open_file = bz2.open if filepath.endswith('bz2') else open
    with open_file(filepath, mode="rb") as file_reader:
        for line in file_reader:
            object_triple = object_pattern.match(line)
            if object_triple:
                [sub, pred, obj] = object_triple.groups()
                sub = urllib.parse.unquote(sub.decode('utf-8'))
                pred = pred.decode('utf-8')
                obj = urllib.parse.unquote(obj.decode('utf-8'))
                yield Triple(sub=sub, pred=pred, obj=obj, is_literal=False)
            else:
                literal_triple = literal_pattern.match(line)
                if literal_triple:
                    [sub, pred, obj] = literal_triple.groups()
                    sub = urllib.parse.unquote(sub.decode('utf-8'))
                    yield Triple(sub=sub, pred=pred.decode('utf-8'), obj=obj.decode('utf-8'), is_literal=True)


def create_set_from_rdf(filepaths: list, valid_pred: RdfPredicate, valid_obj: Optional[str], casting_fn=None) -> set:
    """Create a set of its subjects from a given triple file."""
    data_set = set()
    for fp in filepaths:
        for sub, pred, obj, _ in parse_triples_from_file(fp):
            if pred == valid_pred.value and valid_obj in [None, obj]:
                try:
                    sub, obj = _cast_type(casting_fn, sub, obj, True)
                except KeyError as e:
                    utils.get_logger().debug(str(e))
                    continue
                data_set.add(sub)
    return data_set


def create_multi_val_dict_from_rdf(filepaths: list, valid_pred: RdfPredicate, reverse_key=False, reflexive=False, casting_fn=None) -> dict:
    """Create a key-value dict from a given triple file."""
    data_dict = defaultdict(set)
    for fp in filepaths:
        for sub, pred, obj, is_literal in parse_triples_from_file(fp):
            if pred == valid_pred.value:
                try:
                    sub, obj = _cast_type(casting_fn, sub, obj, is_literal)
                except KeyError as e:
                    utils.get_logger().debug(str(e))
                    continue
                if reflexive or reverse_key:
                    data_dict[obj].add(sub)
                if reflexive or not reverse_key:
                    data_dict[sub].add(obj)
    return data_dict


def create_multi_val_count_dict_from_rdf(filepaths: list, valid_pred: RdfPredicate, reverse_key=False, casting_fn=None) -> dict:
    """Create a key-value dict with frequencies from a given triple file."""
    data_dict = defaultdict(Counter)
    for fp in filepaths:
        for sub, pred, obj, is_literal in parse_triples_from_file(fp):
            if pred == valid_pred.value:
                try:
                    sub, obj = _cast_type(casting_fn, sub, obj, is_literal)
                except KeyError as e:
                    utils.get_logger().debug(str(e))
                    continue
                cleaned_obj = str_util.regularize_spaces(obj.lower())
                if cleaned_obj:
                    if reverse_key:
                        data_dict[cleaned_obj][sub] += 1
                    else:
                        data_dict[sub][cleaned_obj] += 1
    return data_dict


def create_single_val_dict_from_rdf(filepaths: list, valid_pred: RdfPredicate, reverse_key=False, reflexive=False, casting_fn=None) -> dict:
    """Create a key-value mapping from a given triple file."""
    data_dict = {}
    for fp in filepaths:
        for sub, pred, obj, is_literal in parse_triples_from_file(fp):
            if pred == valid_pred.value:
                try:
                    sub, obj = _cast_type(casting_fn, sub, obj, is_literal)
                except KeyError as e:
                    utils.get_logger().debug(str(e))
                    continue
                if reflexive or reverse_key:
                    data_dict[obj] = sub
                elif reflexive or not reverse_key:
                    data_dict[sub] = obj
    return data_dict


def create_dict_from_rdf(filepaths: list, reverse_key=False, casting_fn=None) -> dict:
    """Create a two-dimensional dict from a given triple file."""
    data_dict = defaultdict(functools.partial(defaultdict, set))
    for fp in filepaths:
        for sub, pred, obj, is_literal in parse_triples_from_file(fp):
            try:
                sub, obj = _cast_type(casting_fn, sub, obj, is_literal)
            except KeyError as e:
                utils.get_logger().debug(str(e))
                continue
            data_dict[obj][pred].add(sub) if reverse_key else data_dict[sub][pred].add(obj)
    return data_dict


def _cast_type(casting_fn, sub: str, obj: str, is_literal: bool):
    if casting_fn is not None:
        sub = casting_fn(sub)
        obj = obj if is_literal else casting_fn(obj)
    return sub, obj
