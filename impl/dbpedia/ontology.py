from typing import Set, Dict, Optional, Union
from impl.util.singleton import Singleton
from collections import defaultdict
import networkx as nx
import impl.util.rdf as rdf_util
from impl.util.rdf import Namespace, RdfPredicate, RdfClass, RdfResource
import impl.dbpedia.util as dbp_util
import utils


class DbpClass(RdfResource):
    @classmethod
    def _get_store(cls) -> str:
        return DbpOntologyStore.instance()

    @classmethod
    def get_namespace(cls) -> str:
        return Namespace.DBP_ONTOLOGY.value


class DbpType(DbpClass):
    pass


class DbpPredicate(DbpClass):
    pass


class DbpObjectPredicate(DbpPredicate):
    pass


class DbpDatatypePredicate(DbpPredicate):
    pass


class DbpClassNotExistingException(KeyError):
    pass


@Singleton
class DbpOntologyStore:
    def __init__(self):
        all_classes = utils.load_or_create_cache('dbpedia_classes', self._init_class_cache)
        self.classes_by_idx = {c.idx: c for c in all_classes}
        self.classes_by_name = {c.name: c for c in all_classes}
        self.equivalents = None
        self.labels = None
        # types
        self.type_graph = self._init_type_graph()
        self.transitive_subtypes = {}
        self.transitive_supertypes = {}
        self.main_equivalent_types = None
        self.disjoint_types = None
        self.type_depths = None
        self.type_lexicalisations = None
        self.label_to_types = None
        # predicates
        self.domains = None
        self.ranges = None

    def _init_class_cache(self) -> Set[Union[DbpType, DbpPredicate]]:
        all_classes = {DbpType(0, 'Thing', False)}  # initialize root type with 0
        # predicates
        object_predicate_uris = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.TYPE, RdfClass.OWL_OBJECT_PROPERTY.value)
        object_predicate_uris = {p for p in object_predicate_uris if dbp_util.is_class_iri(p)}
        all_classes.update({DbpObjectPredicate(idx, dbp_util.class_iri2name(uri), False) for idx, uri in enumerate(object_predicate_uris, start=len(all_classes))})
        datatype_predicate_uris = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.TYPE, RdfClass.OWL_DATATYPE_PROPERTY.value)
        datatype_predicate_uris = {p for p in datatype_predicate_uris if dbp_util.is_class_iri(p)}
        all_classes.update({DbpDatatypePredicate(idx, dbp_util.class_iri2name(uri), False) for idx, uri in enumerate(datatype_predicate_uris, start=len(all_classes))})
        # types
        all_type_uris = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.TYPE, RdfClass.OWL_CLASS.value)
        all_type_uris.update(rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.SUBCLASS_OF, None))
        all_type_uris.update(set(rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.DOMAIN, reverse_key=True)))
        all_type_uris.update(set(rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.RANGE, reverse_key=True)))
        all_type_uris.update(set(rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.instance_types')], RdfPredicate.TYPE, reverse_key=True)))
        all_type_uris = {t for t in all_type_uris if dbp_util.is_class_iri(t)}
        all_classes.update({DbpType(idx, dbp_util.class_iri2name(uri), False) for idx, uri in enumerate(all_type_uris, start=len(all_classes))})
        return all_classes

    def _init_type_graph(self) -> nx.DiGraph:
        subtype_mapping = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.SUBCLASS_OF, reverse_key=True, casting_fn=self.get_class_by_iri)
        # completing subtypes with subtypes of equivalent types
        subtype_mapping = {t: {est for et in self.get_equivalents(t) for st in subtype_mapping[et] for est in self.get_equivalents(st)} for t in set(subtype_mapping)}
        return nx.DiGraph(incoming_graph_data=[(t, st) for t, sts in subtype_mapping.items() for st in sts])

    def get_class_by_idx(self, idx: int) -> DbpClass:
        if not self.has_class_with_idx(idx):
            raise DbpClassNotExistingException(f'Could not find class for index: {idx}')
        return self.classes_by_idx[idx]

    def has_class_with_idx(self, idx: int) -> bool:
        return idx in self.classes_by_idx

    def get_class_by_name(self, name: str) -> DbpClass:
        if not self.has_class_with_name(name):
            raise DbpClassNotExistingException(f'Could not find class for name: {name}')
        return self.classes_by_name[name]

    def has_class_with_name(self, name: str) -> bool:
        return name in self.classes_by_name

    def get_class_by_iri(self, iri: str) -> DbpClass:
        if not self.has_class_with_iri(iri):
            raise DbpClassNotExistingException(f'Could not find class for iri: {iri}')
        return self.get_class_by_name(dbp_util.class_iri2name(iri))

    def has_class_with_iri(self, iri: str) -> bool:
        return self.has_class_with_name(dbp_util.class_iri2name(iri))

    def get_equivalents(self, cls: DbpClass) -> set:
        if not self.equivalents:
            self.equivalents = defaultdict(set)
            equivalent_types = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.EQUIVALENT_CLASS, reflexive=True, casting_fn=self.get_class_by_iri)
            for t, ets in equivalent_types.items():
                self.equivalents[t] = ets
            equivalent_predicates = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.EQUIVALENT_PROPERTY, casting_fn=self.get_class_by_iri)
            for p, eps in equivalent_predicates.items():
                self.equivalents[p] = eps
        return {cls} | self.equivalents[cls]

    def get_label(self, cls: DbpClass) -> Optional[str]:
        if not self.labels:
            labels = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.LABEL, casting_fn=self.get_class_by_iri)
            self.labels = defaultdict(lambda: None, labels)
        return self.labels[cls]

    # types

    def get_type_root(self):
        return self.get_class_by_idx(0)  # root is initialized as 0

    def get_types(self, include_root=True) -> Set[DbpType]:
        types = {c for c in self.classes_by_idx.values() if isinstance(c, DbpType)}
        return types if include_root else types.difference({self.get_type_root()})

    def get_types_for_label(self, label: str) -> Set[DbpType]:
        if self.label_to_types is None:
            self.label_to_types = defaultdict(set)
            for t in self.get_types(include_root=False):
                self.label_to_types[self.get_label(t).lower().split()[-1]].add(t)
        return self.label_to_types[label.lower()]

    def get_independent_types(self, dbp_types: Set[DbpType], exclude_root: bool = False) -> Set[DbpType]:
        independent_types = dbp_types.difference({st for t in dbp_types for st in self.get_transitive_supertypes(t)})
        return independent_types.difference({self.get_type_root()}) if exclude_root else independent_types

    def get_supertypes(self, t: DbpType) -> Set[DbpType]:
        return set(self.type_graph.predecessors(t)) if t in self.type_graph else set()

    def get_transitive_supertypes(self, t: DbpType, include_root=True, include_self=False) -> Set[DbpType]:
        if t not in self.transitive_supertypes:
            self.transitive_supertypes[t] = nx.ancestors(self.type_graph, t) if t in self.type_graph else set()
        tt = self.transitive_supertypes[t]
        tt = tt if include_root else tt.difference({self.get_type_root()})
        tt = tt | {t} if include_self else tt
        return tt

    def get_subtypes(self, t: DbpType) -> Set[DbpType]:
        return set(self.type_graph.successors(t)) if t in self.type_graph else set()

    def get_transitive_subtypes(self, t: DbpType, include_self=False) -> Set[DbpType]:
        if t not in self.transitive_subtypes:
            self.transitive_subtypes[t] = nx.descendants(self.type_graph, t) if t in self.type_graph else set()
        tt = self.transitive_subtypes[t]
        tt = tt | {t} if include_self else tt
        return tt

    def are_equivalent_types(self, dbp_types: Set[DbpType]) -> bool:
        """Return True, if types are equivalent."""
        return dbp_types.issubset(self.get_equivalents(list(dbp_types)[0]))

    def get_main_equivalence_type(self, t: DbpType) -> str:
        """Return the main equivalence type (i.e. the most strongly linked type in the taxonomy) for a given type."""
        if self.main_equivalent_types is None:
            self.main_equivalent_types = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.SUBCLASS_OF, None, casting_fn=self.get_class_by_iri)
            self.main_equivalent_types.add(self.get_type_root())
        return self.get_equivalents(t).intersection(self.main_equivalent_types).pop()

    def get_disjoint_types(self, t: DbpType) -> Set[DbpType]:
        """Return all types that are disjoint with `dbp_type` (excluding the wrong disjointness Agent<->Place)."""
        if self.disjoint_types is None:
            # find all disjoint types
            disjoint_types = defaultdict(set, rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.DISJOINT_WITH, reflexive=True, casting_fn=self.get_class_by_iri))
            # correct some mistakes manually
            removed_axioms = [('Agent', 'Place')]
            for a, b in removed_axioms:
                a, b = self.get_class_by_name(a), self.get_class_by_name(b)
                disjoint_types[a].remove(b)
                disjoint_types[b].remove(a)
            added_axioms = [('Person', 'Place'), ('Family', 'Place')]
            for a, b in added_axioms:
                a, b = self.get_class_by_name(a), self.get_class_by_name(b)
                disjoint_types[a].add(b)
                disjoint_types[b].add(a)
            # complete the subtype of each type with the subtypes of its disjoint types
            disjoint_types = {t: {st for dt in dts for st in self.get_transitive_subtypes(dt, include_self=True)} for t, dts in disjoint_types.items()}
            self.disjoint_types = defaultdict(set, disjoint_types)
        return self.disjoint_types[t]

    def get_type_depth(self, t: DbpType) -> int:
        """Return the shortest way from `t` to the root of the type graph."""
        if self.type_depths is None:
            self.type_depths = defaultdict(lambda: 1, nx.shortest_path_length(self.type_graph, source=self.get_type_root()))
        return self.type_depths[t]

    def get_type_lexicalisations(self, lemma: str) -> Dict[DbpType, int]:
        """Return the type lexicalisation score for a set of lemmas (i.e. the probabilities of types given `lemmas`)."""
        if self.type_lexicalisations is None:
            self.type_lexicalisations = defaultdict(dict, utils.load_cache('wikipedia_type_lexicalisations'))
        return self.type_lexicalisations[lemma.lower()]

    # predicates

    def get_predicates(self) -> Set[DbpPredicate]:
        return {c for c in self.classes_by_idx.values() if isinstance(c, DbpPredicate)}

    def get_domain(self, pred: DbpPredicate) -> Optional[DbpType]:
        if self.domains is None:
            domains = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.DOMAIN, casting_fn=self.get_class_by_iri)
            self.domains = defaultdict(lambda: None, domains)
        return self.domains[pred]

    def get_range(self, pred: DbpPredicate) -> Optional[DbpType]:
        if self.ranges is None:
            ranges = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.taxonomy')], RdfPredicate.RANGE, casting_fn=self.get_class_by_iri)
            self.ranges = defaultdict(lambda: None, ranges)
        return self.ranges[pred]
