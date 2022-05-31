from typing import Set, Dict, Union, Tuple, Any, List
from collections import defaultdict, Counter
import utils
from impl.util.singleton import Singleton
from impl.util.rdf import Namespace, RdfResource, EntityIndex
from impl.dbpedia.ontology import DbpType, DbpOntologyStore
from impl.dbpedia.resource import DbpResource, DbpEntity, DbpResourceStore
from impl.dbpedia.category import DbpCategory, DbpCategoryStore
from impl import subject_entity
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

    def get_types(self) -> Set[ClgType]:
        return self._get_store().get_types(self)

    def get_transitive_types(self, include_root=False) -> Set[ClgType]:
        return self._get_store().get_transitive_types(self, include_root=include_root)

    def get_independent_types(self) -> Set[ClgType]:
        return self._get_store().get_independent_types(self)

    def get_all_dbp_types(self, add_transitive_closure=False) -> Set[DbpType]:
        return {dt for ct in self.get_types() for dt in ct.get_all_dbp_types(add_transitive_closure=add_transitive_closure)}

    def get_properties(self, as_tuple=False) -> Dict[ClgPredicate, set]:
        return self._get_store().get_properties(self, as_tuple=as_tuple)

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
        self.dbo = DbpOntologyStore.instance()
        self.dbr = DbpResourceStore.instance()
        self.clgo = ClgOntologyStore.instance()

        all_entities, page_occurrence_data = utils.load_or_create_cache('caligraph_entities', self._init_entity_cache)
        self.entities_by_idx = {e.idx: e for e in all_entities}
        self.entities_by_name = {e.name: e for e in all_entities}
        self.page_occurrence_data = {self.entities_by_idx[idx]: data for idx, data in page_occurrence_data.items()}
        self.labels = self._init_labels()
        self.surface_forms = self._init_surface_forms()
        self.types = self._init_types()
        self.properties = self._init_properties()
        self.provenance_resources = defaultdict(set)
        self.entity_stats = None

        self.axioms = None
        self.axiom_properties = None

    def _init_entity_cache(self) -> Tuple[List[ClgEntity], Dict[int, Set[Tuple[int, str, int, str, int, str, str]]]]:
        # incorporate existing entities from DBpedia (but no redirects or disambiguations)
        all_entities = [ClgEntity(e.idx, e.name, False) for e in self.dbr.get_entities()]
        max_idx = max(e.idx for e in all_entities)
        # initialize new entities from subject entities on pages
        page_occurrence_data = defaultdict(set)
        for res, entities_per_ts in subject_entity.get_page_subject_entities().items():
            for ts, entities_per_s in entities_per_ts.items():
                for s, entities in entities_per_s.items():
                    for ent_name, ent_data in entities.items():
                        ts_ent = ent_data['TS_entidx'] if ent_data['TS_entidx'] is not None else EntityIndex.NO_ENTITY.value
                        s_ent = ent_data['S_entidx'] if ent_data['S_entidx'] is not None else EntityIndex.NO_ENTITY.value
                        page_occurrence_data[ent_name].add((res.idx, ts, ts_ent, s, s_ent, ent_data['text'], ent_data['tag']))
        new_entity_names = set(page_occurrence_data).difference({e.name for e in all_entities})
        for idx, ent_name in enumerate(new_entity_names, start=max_idx + 1):
            all_entities.append(ClgEntity(idx, ent_name, False))
        # update page occurrence data with actual entity indices
        entity_name_to_index = {e.name: e.idx for e in all_entities}
        page_occurrence_data = {entity_name_to_index[ent_name]: data for ent_name, data in page_occurrence_data.items()}
        return all_entities, page_occurrence_data

    def add_axiom_information(self, axiom_information: Dict[ClgType, Set[Tuple[ClgPredicate, Any, float]]]):
        self.entity_stats = None  # reset, as we are adding new entity information

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
        self.entity_stats = None  # reset, as we are adding new entity information

        for ent_idx, origin_data in listing_information.items():
            for (res_idx, section_name), data in origin_data.items():
                ent = self.get_entity_by_idx(ent_idx)
                self.provenance_resources[ent].add(self.dbr.get_resource_by_idx(res_idx))
                self.surface_forms[ent].update(data['labels'])
                self.types[ent].update({self.clgo.get_class_by_idx(tidx) for tidx in data['types']})
                for p, v in data['out']:
                    pred = self.clgo.get_class_by_idx(p)
                    val = self.get_entity_by_idx(v)
                    self.properties[ent][pred].add(val)
                for p, s in data['in']:
                    pred = self.clgo.get_class_by_idx(p)
                    sub = self.get_entity_by_idx(s)
                    self.properties[sub][pred].add(ent)

    def get_page_occurrence_data(self) -> Dict[ClgEntity, Set[Tuple[int, str, int, str, int, str, str]]]:
        return self.page_occurrence_data

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

    def _init_labels(self) -> Dict[ClgEntity, str]:
        labels = {}
        for dbp_ent in self.dbr.get_entities():
            labels[self.get_entity_for_dbp_entity(dbp_ent)] = dbp_ent.get_label()
        for ent, data_tuples in self.get_page_occurrence_data().items():
            if ent not in labels:
                label_counter = Counter([t[5] for t in data_tuples])
                labels[ent] = label_counter.most_common(1)[0][0]
        return labels

    def get_label(self, ent: ClgEntity) -> str:
        return self.labels[ent]

    def _init_surface_forms(self) -> Dict[ClgEntity, Set[str]]:
        surface_forms = defaultdict(set)
        for dbp_ent in self.dbr.get_entities():
            surface_forms[self.get_entity_for_dbp_entity(dbp_ent)].update(dbp_ent.get_surface_forms())
        for ent, data_tuples in self.get_page_occurrence_data().items():
            surface_forms[ent].update({t[5] for t in data_tuples})
        return surface_forms

    def get_surface_forms(self, ent: ClgEntity) -> Set[str]:
        return self.surface_forms[ent]

    def _init_types(self) -> Dict[ClgEntity, Set[ClgType]]:
        types = defaultdict(set)
        # retrieve types based on Wikipedia category membership
        for cat in DbpCategoryStore.instance().get_categories(include_listcategories=True):
            cat_types = self.clgo.get_types_for_associated_dbp_resource(cat)
            if cat_types:
                for dbp_ent in cat.get_entities():
                    types[self.get_entity_for_dbp_entity(dbp_ent)].update(cat_types)
        # retrieve types based on Wikipedia listpage membership
        listpages_by_idx = {lp.idx: lp for lp in self.dbr.get_listpages()}
        for ent, data_tuples in self.get_page_occurrence_data().items():
            for dt in data_tuples:
                res_idx = dt[0]
                if res_idx in listpages_by_idx:
                    lp_types = self.clgo.get_types_for_associated_dbp_resource(listpages_by_idx[res_idx])
                    types[ent].update(lp_types)
        # retrieve types based on DBpedia types
        for dbp_ent in self.dbr.get_entities():
            ent = self.get_entity_for_dbp_entity(dbp_ent)
            types_from_dbpedia = {ct for dt in dbp_ent.get_types() for ct in self.clgo.get_types_for_associated_dbp_type(dt)}
            types[ent].update(types_from_dbpedia)
            # discard types that are not in accordance with the DBpedia types of the entity
            disjoint_types = {dt for ct in types_from_dbpedia for dt in self.clgo.get_disjoint_types(ct)}
            types[ent].difference_update(disjoint_types)
        # remove potential transitive types
        for ent, ent_types in types.items():
            types[ent].difference_update({tt for ct in ent_types for tt in self.clgo.get_transitive_supertypes(ct)})
        # remove remaining disjointnesses in types
        for ent, ent_types in types.items():
            types[ent].difference_update({dt for ct in ent_types for dt in self.clgo.get_disjoint_types(ct)})
        return types

    def get_types(self, ent: ClgEntity) -> Set[ClgType]:
        return self.types[ent]

    def get_transitive_types(self, ent: ClgEntity, include_root=False) -> Set[ClgType]:
        return {tt for t in self.types[ent] for tt in self.clgo.get_transitive_supertypes(t, include_root=include_root, include_self=True)}

    def get_independent_types(self, ent: ClgEntity) -> Set[ClgType]:
        return self.clgo.get_independent_types(self.get_types(ent))

    def _init_properties(self) -> Dict[ClgEntity, Dict[ClgPredicate, set]]:
        properties = defaultdict(lambda: defaultdict(set))
        for dbp_ent, props in self.dbr.get_entity_properties().items():
            ent = self.get_entity_for_dbp_entity(dbp_ent)
            for dbp_pred, vals in props.items():
                pred = self.clgo.get_predicate_for_dbp_predicate(dbp_pred)
                if isinstance(pred, ClgObjectPredicate):
                    vals = {self.get_entity_for_dbp_entity(v) for v in vals if self.has_entity_for_dbp_entity(v)}
                properties[ent][pred].update(vals)
        return properties

    def get_properties(self, ent: ClgEntity, as_tuple=False) -> Union[Dict[ClgPredicate, set], Set[Tuple[ClgPredicate, Any]]]:
        return {(p, v) for p, vals in self.properties[ent].items() for v in vals} if as_tuple else self.properties[ent]

    def get_entity_properties(self) -> Dict[ClgEntity, Dict[ClgPredicate, set]]:
        return {e: self.get_properties(e) for e in self.get_entities()}

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
