from typing import Set, Dict, Union, Tuple, Any, List, Optional
from collections import defaultdict, Counter
from functools import partial
import utils
from impl.util.singleton import Singleton
from impl.util.rdf import Namespace, RdfResource
from impl.util.nlp import EntityTypeLabel
from impl.wikipedia import WikiPageStore
from impl.dbpedia.ontology import DbpType
from impl.dbpedia.resource import DbpResource, DbpEntity, DbpResourceStore, DbpListpage
from impl.dbpedia.category import DbpCategory, DbpCategoryStore
from impl.caligraph.ontology import ClgType, ClgPredicate, ClgObjectPredicate, ClgOntologyStore


class ClgEntity(RdfResource):
    def has_dbp_entity(self) -> bool:
        return self._get_store().has_dbp_entity(self)

    def get_dbp_entity(self) -> DbpEntity:
        return self._get_store().get_dbp_entity(self)

    def get_provenance_resources(self) -> Set[Union[DbpResource, DbpCategory]]:
        return self._get_store().get_provenance_resources(self)

    def get_surface_forms(self) -> Set[str]:
        return self._get_store().get_surface_forms(self)

    def get_abstract(self) -> Optional[str]:
        if not self.has_dbp_entity():
            return None
        return self.get_dbp_entity().get_abstract()

    def get_types(self) -> Set[ClgType]:
        return self._get_store().get_types(self)

    def get_type_label(self) -> EntityTypeLabel:
        return self._get_store().dbr.get_type_label(self.idx)

    def get_transitive_types(self, include_root=False) -> Set[ClgType]:
        return self._get_store().get_transitive_types(self, include_root=include_root)

    def get_independent_types(self) -> Set[ClgType]:
        return self._get_store().get_independent_types(self)

    def get_all_dbp_types(self, add_transitive_closure=False) -> Set[DbpType]:
        return {dt for ct in self.get_types() for dt in ct.get_all_dbp_types(add_transitive_closure=add_transitive_closure)}

    def get_properties(self, as_tuple=False) -> Dict[ClgPredicate, set]:
        return self._get_store().get_properties(self, as_tuple=as_tuple)

    def get_inverse_properties(self, as_tuple=False) -> Dict[ClgPredicate, set]:
        return self._get_store().get_inverse_properties(self, as_tuple=as_tuple)

    def get_axiom_properties(self) -> Dict[ClgPredicate, set]:
        return self._get_store().get_axiom_properties(self)

    @classmethod
    def _get_store(cls):
        return ClgEntityStore.instance()

    @classmethod
    def get_namespace(cls) -> str:
        return Namespace.CLG_RESOURCE.value


class ClgEntityNotExistingException(KeyError):
    pass


@Singleton
class ClgEntityStore:
    def __init__(self):
        self.dbr = DbpResourceStore.instance()
        self.clgo = ClgOntologyStore.instance()

        all_entities = utils.load_or_create_cache('caligraph_entities', self._init_entity_cache)
        self.entities_by_idx = {e.idx: e for e in all_entities}
        self.entities_by_name = {e.name: e for e in all_entities}

        self.new_entities = set()
        self.labels = None
        self.surface_forms = None
        self.types = None
        self.properties = None
        self.inverse_properties = None
        self.provenance_resources = defaultdict(set)
        self.entity_stats = None

        self.axioms = None
        self.axiom_properties = None

    def _reset_precomputed_attributes(self):
        """Reset precomputed attributes in case new information is added to the store."""
        self.inverse_properties = None
        self.entity_stats = None

    def _init_entity_cache(self) -> List[ClgEntity]:
        # incorporate existing entities from DBpedia (but no redirects or disambiguations)
        return [ClgEntity(e.idx, e.name, False) for e in self.dbr.get_entities()]

    def add_subject_entities(self):
        self._load_labels()
        self._load_surface_forms()
        self._load_types()
        self._reset_precomputed_attributes()
        subject_entity_info = defaultdict(lambda: {'labels': Counter(), 'types': set(), 'provenance': set()})
        # collect info about subject entities
        for wp in WikiPageStore.instance().get_pages():
            types = self.clgo.get_types_for_associated_dbp_resource(wp.resource) if isinstance(wp.resource, DbpListpage) else set()
            for se_mention in wp.get_subject_entities():
                idx = se_mention.entity_idx
                subject_entity_info[idx]['labels'][se_mention.label] += 1
                subject_entity_info[idx]['types'].update(types)
                subject_entity_info[idx]['provenance'].add(wp.resource)
        # add info to store
        for ent_idx, ent_info in subject_entity_info.items():
            if self.has_entity_with_idx(ent_idx):
                ent = self.get_entity_by_idx(ent_idx)
            else:
                # add entity and its label if not existing
                ent_label = ent_info['labels'].most_common(1)[0][0]
                # TODO: for now, we ensure a unique entity name by appending its IDX => find more expressive name!
                ent_name = f'{ent_label}_({ent_idx})'
                ent = ClgEntity(ent_idx, ent_name, False)
                self.entities_by_idx[ent_idx] = ent
                self.entities_by_name[ent_name] = ent
                self.new_entities.add(ent_idx)
                self.labels[ent] = ent_label
            self.surface_forms[ent].update({label for label, cnt in ent_info['labels'].items() if cnt > 2})
            self.provenance_resources[ent] = self.get_provenance_resources(ent) | ent_info['provenance']
            self.types[ent].update(ent_info['types'])
        self._clean_types(self.types)

    def add_axiom_information(self, axiom_information: Dict[ClgType, Set[Tuple[ClgPredicate, Any, float]]]):
        self._load_properties()
        self._reset_precomputed_attributes()
        self.axioms = defaultdict(set, {ct: {tuple(axiom[1:3]) for axiom in axioms} for ct, axioms in axiom_information.items()})
        # remove redundant axioms by first applying full transitivity to the axioms and then reducing bottom-up
        for node in self.clgo.graph.traverse_nodes_topdown():
            clg_type = self.clgo.get_class_by_name(node)
            for subtype in self.clgo.get_subtypes(clg_type):
                self.axioms[subtype].update(self.axioms[clg_type])
        for node in self.clgo.graph.traverse_nodes_bottomup():
            clg_type = self.clgo.get_class_by_name(node)
            supertype_axioms = {a for st in self.clgo.get_supertypes(clg_type) for a in self.axioms[st]}
            self.axioms[clg_type].difference_update(supertype_axioms)
        # extract actual properties for entities from axioms
        self.axiom_properties = defaultdict(lambda: defaultdict(set))
        for ent in self.get_entities():
            for t in ent.get_transitive_types():
                for pred, val in self.axioms[t]:
                    self.axiom_properties[ent][pred].add(val)
                    self.properties[ent][pred].add(val)

    def add_listing_information(self, listing_information: Dict[int, Dict[Tuple[int, str], dict]]):
        self._load_properties()
        self._reset_precomputed_attributes()
        for ent_idx, origin_data in listing_information.items():
            ent = self.get_entity_by_idx(ent_idx)
            disjoint_ent_types = {dt for t in ent.get_types() for dt in self.clgo.get_disjoint_types(t)}
            for (res_idx, section_idx), data in origin_data.items():
                new_types = {self.clgo.get_class_by_idx(tidx) for tidx in data['types']}
                if new_types.intersection(disjoint_ent_types):
                    continue  # skip the origin for this entity if conflicts are discovered
                self.provenance_resources[ent].add(self.dbr.get_resource_by_idx(res_idx))
                self.surface_forms[ent].update(data['labels'])
                self.types[ent].update(new_types)
                for p, v in data['out']:
                    pred = self.clgo.get_class_by_idx(p)
                    val = self.get_entity_by_idx(v)
                    self.properties[ent][pred].add(val)
                for p, s in data['in']:
                    pred = self.clgo.get_class_by_idx(p)
                    sub = self.get_entity_by_idx(s)
                    self.properties[sub][pred].add(ent)
        self._clean_types(self.types)

    def get_entities(self) -> Set[ClgEntity]:
        return set(self.entities_by_idx.values())

    def has_entity_with_idx(self, idx: int) -> bool:
        return idx in self.entities_by_idx

    def get_entity_by_idx(self, idx: int) -> ClgEntity:
        if not self.has_entity_with_idx(idx):
            raise ClgEntityNotExistingException(f'Could not find resource for index: {idx}')
        return self.entities_by_idx[idx]

    def has_entity_with_name(self, name: str) -> bool:
        return name in self.entities_by_name

    def get_entity_by_name(self, name: str) -> ClgEntity:
        if not self.has_entity_with_name(name):
            raise ClgEntityNotExistingException(f'Could not find resource for name: {name}')
        return self.entities_by_name[name]

    def has_entity_for_dbp_entity(self, dbp_ent: DbpEntity) -> bool:
        dbp_ent = self.dbr.resolve_spelling_redirect(dbp_ent)
        return self.has_entity_with_idx(dbp_ent.idx)

    def get_entity_for_dbp_entity(self, dbp_ent: DbpEntity) -> ClgEntity:
        dbp_ent = self.dbr.resolve_spelling_redirect(dbp_ent)
        return self.get_entity_by_idx(dbp_ent.idx)

    def has_dbp_entity(self, ent: ClgEntity) -> bool:
        return self.dbr.has_resource_with_idx(ent.idx)

    def get_dbp_entity(self, ent: ClgEntity) -> DbpEntity:
        return self.dbr.get_resource_by_idx(ent.idx)

    def get_provenance_resources(self, ent: ClgEntity) -> Set[Union[DbpResource, DbpCategory]]:
        if ent not in self.provenance_resources:
            self.provenance_resources[ent].update({r for t in self.types[ent] for r in t.get_associated_dbp_resources()})
        return self.provenance_resources[ent]

    def get_label(self, ent: ClgEntity) -> str:
        self._load_labels()
        return self.labels[ent]

    def _load_labels(self):
        if self.labels is None:
            self.labels = utils.load_or_create_cache('caligraph_entity_labels', self._init_labels)

    def _init_labels(self) -> Dict[ClgEntity, str]:
        labels = {}
        for dbp_ent in self.dbr.get_entities():
            labels[self.get_entity_for_dbp_entity(dbp_ent)] = dbp_ent.get_label()
        return labels

    def get_surface_forms(self, ent: ClgEntity) -> Set[str]:
        self._load_surface_forms()
        return self.surface_forms[ent]

    def _load_surface_forms(self):
        if self.surface_forms is None:
            self.surface_forms = utils.load_or_create_cache('caligraph_entity_surface_forms', self._init_surface_forms)

    def _init_surface_forms(self) -> Dict[ClgEntity, Set[str]]:
        surface_forms = defaultdict(set)
        for dbp_ent in self.dbr.get_entities():
            surface_forms[self.get_entity_for_dbp_entity(dbp_ent)].update(dbp_ent.get_surface_forms())
        return surface_forms

    def get_types(self, ent: ClgEntity) -> Set[ClgType]:
        self._load_types()
        return self.types[ent]

    def _load_types(self):
        if self.types is None:
            self.types = utils.load_or_create_cache('caligraph_entity_types', self._init_types)

    def _init_types(self) -> Dict[ClgEntity, Set[ClgType]]:
        types = defaultdict(set)
        # retrieve types based on Wikipedia category membership
        for cat in DbpCategoryStore.instance().get_categories(include_listcategories=True):
            cat_types = self.clgo.get_types_for_associated_dbp_resource(cat)
            if cat_types:
                for dbp_ent in cat.get_entities():
                    types[self.get_entity_for_dbp_entity(dbp_ent)].update(cat_types)
        # retrieve types based on DBpedia types
        for dbp_ent in self.dbr.get_entities():
            ent = self.get_entity_for_dbp_entity(dbp_ent)
            types_from_dbpedia = {ct for dt in dbp_ent.get_types() for ct in self.clgo.get_types_for_associated_dbp_type(dt)}
            types[ent].update(types_from_dbpedia)
            # discard types that are not in accordance with the DBpedia types of the entity
            disjoint_types = {dt for ct in types_from_dbpedia for dt in self.clgo.get_disjoint_types(ct)}
            types[ent].difference_update(disjoint_types)
        self._clean_types(types)
        return types

    def _clean_types(self, types: Dict[ClgEntity, Set[ClgType]]):
        # remove potential transitive types
        for ent, ent_types in types.items():
            types[ent].difference_update({tt for ct in ent_types for tt in self.clgo.get_transitive_supertypes(ct)})
        # remove remaining disjointnesses in types
        for ent, ent_types in types.items():
            types[ent].difference_update({dt for ct in ent_types for dt in self.clgo.get_disjoint_types(ct)})

    def get_transitive_types(self, ent: ClgEntity, include_root=False) -> Set[ClgType]:
        return {tt for t in self.get_types(ent) for tt in self.clgo.get_transitive_supertypes(t, include_root=include_root, include_self=True)}

    def get_independent_types(self, ent: ClgEntity) -> Set[ClgType]:
        return self.clgo.get_independent_types(self.get_types(ent))

    def get_properties(self, ent: ClgEntity, as_tuple=False) -> Union[Dict[ClgPredicate, set], Set[Tuple[ClgPredicate, Any]]]:
        self._load_properties()
        return {(p, v) for p, vals in self.properties[ent].items() for v in vals} if as_tuple else self.properties[ent]

    def _load_properties(self):
        if self.properties is None:
            self.properties = utils.load_or_create_cache('caligraph_entity_properties', self._init_properties)

    def _init_properties(self) -> Dict[ClgEntity, Dict[ClgPredicate, set]]:
        properties = defaultdict(partial(defaultdict, set))
        for dbp_ent, props in self.dbr.get_entity_properties().items():
            ent = self.get_entity_for_dbp_entity(dbp_ent)
            for dbp_pred, vals in props.items():
                pred = self.clgo.get_predicate_for_dbp_predicate(dbp_pred)
                if isinstance(pred, ClgObjectPredicate):
                    vals = {self.get_entity_for_dbp_entity(v) for v in vals if self.has_entity_for_dbp_entity(v)}
                properties[ent][pred].update(vals)
        return properties

    def get_entity_properties(self) -> Dict[ClgEntity, Dict[ClgPredicate, set]]:
        return {e: self.get_properties(e) for e in self.get_entities()}

    def get_inverse_properties(self, ent: ClgEntity, as_tuple=False) -> Union[Dict[ClgPredicate, Set[ClgEntity]], Set[Tuple[ClgPredicate, ClgEntity]]]:
        if self.inverse_properties is None:
            self.inverse_properties = defaultdict(lambda: defaultdict(set))
            for sub, props in self.properties.items():
                for pred, vals in props.items():
                    for val in vals:
                        if not isinstance(val, ClgEntity):
                            continue
                        self.inverse_properties[val][pred].add(sub)
        return {(p, v) for p, vals in self.inverse_properties[ent].items() for v in vals} if as_tuple else self.inverse_properties[ent]

    def get_property_frequencies(self, clg_type: ClgType) -> Dict[Tuple[ClgPredicate, Any], float]:
        if self.entity_stats is None:
            self.entity_stats = defaultdict(lambda: {'entity_count': 0, 'property_counts': Counter()})
            for ent in self.get_entities():
                for t in ent.get_types():
                    self.entity_stats[t]['entity_count'] += 1
                    for prop in ent.get_properties(as_tuple=True):
                        self.entity_stats[t]['property_counts'][prop] += 1
            for node in self.clgo.graph.traverse_nodes_bottomup():
                t = self.clgo.get_class_by_name(node)
                self.entity_stats[t]['transitive_entity_count'] = self.entity_stats[t]['entity_count']
                self.entity_stats[t]['transitive_property_counts'] = self.entity_stats[t]['property_counts'].copy()
                if self.entity_stats[t]['entity_count'] < 5:
                    for subtype in self.clgo.get_subtypes(t):
                        self.entity_stats[t]['transitive_entity_count'] += self.entity_stats[subtype]['transitive_entity_count']
                        self.entity_stats[t]['transitive_property_counts'] += self.entity_stats[subtype]['transitive_property_counts']
        stats = self.entity_stats[clg_type]
        entity_count = stats['entity_count']
        property_counts = stats['property_counts']
        if entity_count < 5:
            entity_count = stats['transitive_entity_count']
            property_counts = stats['transitive_property_counts']
        return {p: count / entity_count for p, count in property_counts.items()}

    def get_all_axioms(self) -> Dict[ClgType, Set[Tuple[ClgPredicate, Any]]]:
        return self.axioms

    def get_axioms(self, clg_type: ClgType, transitive=False) -> Set[Tuple[ClgPredicate, Any]]:
        if transitive:
            return {a for t in self.clgo.get_transitive_supertypes(clg_type, include_self=True) for a in self.axioms[t]}
        return self.axioms[clg_type]

    def get_axiom_properties(self, ent: ClgEntity) -> Dict[ClgPredicate, set]:
        return self.axiom_properties[ent]
