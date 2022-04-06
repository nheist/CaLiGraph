from functools import cache
from impl.util.singleton import Singleton
import impl.util.rdf as rdf_util
from impl.util.rdf import Namespace
from impl.util.rdf import RdfPredicate, RdfResource
import impl.dbpedia.util as dbp_util
from impl.dbpedia.ontology import DbpType, DbpPredicate, DbpObjectPredicate, DbpOntologyStore
from polyleven import levenshtein
from typing import Union, Dict, Optional, Set, Any, Tuple
import utils
from collections import defaultdict


class DbpResource(RdfResource):
    def get_surface_forms(self) -> Set[str]:
        return self._get_store().get_surface_forms(self)

    def get_types(self) -> Set[DbpType]:
        return self._get_store().get_types(self)

    def get_independent_types(self) -> Set[DbpType]:
        return self._get_store().get_independent_types(self.get_types())

    def get_transitive_types(self, include_root=False) -> Set[DbpType]:
        return self._get_store().get_transitive_types(self, include_root=include_root)

    def get_properties(self, as_tuple=False) -> Union[Dict[DbpPredicate, set], Set[Tuple[DbpPredicate, Any]]]:
        return self._get_store().get_properties(self, as_tuple)

    @classmethod
    def get_namespace(cls) -> str:
        return Namespace.DBP_RESOURCE.value

    @classmethod
    def _get_store(cls):
        return DbpResourceStore.instance()

    @classmethod
    def _get_prefix(cls) -> str:
        return ''


class DbpEntity(DbpResource):
    pass


class DbpListpage(DbpResource):
    def get_label(self) -> str:
        label = super().get_label()
        label = label[4:] if label.startswith('the ') else label
        return label

    @classmethod
    def _get_prefix(cls) -> str:
        return Namespace.PREFIX_LIST.value


class DbpFile(DbpResource):
    @classmethod
    def _get_prefix(cls) -> str:
        return Namespace.PREFIX_FILE.value


class DbpResourceNotExistingException(KeyError):
    pass


@Singleton
class DbpResourceStore:
    def __init__(self):
        self.dbo = DbpOntologyStore.instance()

        all_resources = utils.load_or_create_cache('dbpedia_resources', self._init_resource_cache)
        self.resources_by_idx = {r.idx: r for r in all_resources}
        self.resources_by_name = {r.name: r for r in all_resources}

        self.labels = None
        self.surface_forms = None
        self.surface_form_references = None

        self.types = None
        self.entities_of_type = None
        self.wikilinks = None
        self.properties = None
        self.inverse_properties = None
        self.redirects = None

    def _init_resource_cache(self) -> Set[DbpResource]:
        # find all resources that have at least a label or a type
        resources_with_label = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.labels')], RdfPredicate.LABEL, None)
        resources_with_type = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.instance_types')], RdfPredicate.TYPE, None)
        all_resource_uris = {dbp_util.get_canonical_iri(uri) for uri in resources_with_label | resources_with_type}  # make all uris canonical
        all_resource_uris = {uri for uri in all_resource_uris if dbp_util.is_resource_iri(uri) and not dbp_util.is_category_iri(uri)}  # filter out invalid uris
        # find resources that are redirects or disambiguations
        meta_resources = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.redirects')], RdfPredicate.REDIRECTS, None)
        meta_resources.update(rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.disambiguations')], RdfPredicate.DISAMBIGUATES, None))

        all_resources = set()
        for idx, res_uri in enumerate(all_resource_uris):
            res_name = dbp_util.resource_iri2name(res_uri)
            is_meta = res_uri in meta_resources
            if dbp_util.is_listpage_iri(res_uri):
                all_resources.add(DbpListpage(idx, res_name, is_meta))
            elif dbp_util.is_file_iri(res_uri):
                all_resources.add(DbpFile(idx, res_name, is_meta))
            else:
                all_resources.add(DbpEntity(idx, res_name, is_meta))
        return all_resources

    def has_resource_with_idx(self, idx: int) -> bool:
        return idx in self.resources_by_idx

    def get_resource_by_idx(self, idx: int) -> DbpResource:
        if not self.has_resource_with_idx(idx):
            raise DbpResourceNotExistingException(f'Could not find resource for index: {idx}')
        return self.resources_by_idx[idx]

    def has_resource_with_name(self, name: str) -> bool:
        return name in self.resources_by_name

    def get_resource_by_name(self, name: str) -> DbpResource:
        if not self.has_resource_with_name(name):
            raise DbpResourceNotExistingException(f'Could not find resource for name: {name}')
        return self.resources_by_name[name]

    def has_resource_with_iri(self, iri: str) -> bool:
        return self.has_resource_with_name(dbp_util.resource_iri2name(iri))

    def get_resource_by_iri(self, iri: str) -> DbpResource:
        try:
            return self.get_resource_by_name(dbp_util.resource_iri2name(iri))
        except DbpResourceNotExistingException:
            raise DbpResourceNotExistingException(f'Could not find resource for iri: {iri}')

    def get_resources(self) -> Set[DbpResource]:
        return set(self.resources_by_idx.values())

    def get_entities(self) -> Set[DbpEntity]:
        return {r for r in self.resources_by_idx.values() if isinstance(r, DbpEntity)}

    def get_listpages(self) -> Set[DbpListpage]:
        return {r for r in self.resources_by_idx.values() if isinstance(r, DbpListpage) and not r.is_meta}

    def get_label(self, res: DbpResource) -> Optional[str]:
        if self.labels is None:
            self.labels = defaultdict(lambda: None, utils.load_or_create_cache('dbpedia_resource_labels', self._init_label_cache))
        return self.labels[res.idx]

    def _init_label_cache(self) -> Dict[int, str]:
        labels = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.labels')], RdfPredicate.LABEL, casting_fn=self.get_resource_by_iri)
        labels = {r.idx: label for r, label in labels.items()}
        return labels

    def get_surface_forms(self, res: DbpResource) -> Set[str]:
        if self.surface_forms is None:
            self.surface_forms = utils.load_or_create_cache('dbpedia_resource_surface_forms', self._init_surface_form_cache)
        return self.surface_forms[res.idx]

    def _init_surface_form_cache(self) -> Dict[int, Set[str]]:
        surface_forms = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.anchor_texts')], RdfPredicate.ANCHOR_TEXT, casting_fn=self.get_resource_by_iri)
        surface_forms = {r.idx: sfs for r, sfs in surface_forms.items()}
        return surface_forms

    def get_surface_form_references(self, text: str) -> Dict[DbpResource, float]:
        if self.surface_form_references is None:
            self.surface_form_references = defaultdict(dict, utils.load_or_create_cache('dbpedia_resource_surface_form_references', self._init_surface_form_reference_cache))
        sf_references = self.surface_form_references[text.lower()]
        sf_references = {self.get_resource_by_idx(res_idx): freq for res_idx, freq in sf_references.items()}
        return sf_references

    def _init_surface_form_reference_cache(self) -> Dict[str, Dict[int, float]]:
        # count how often a lexicalisation points to a given resource
        sf_reference_counts = rdf_util.create_multi_val_count_dict_from_rdf([utils.get_data_file('files.dbpedia.anchor_texts')], RdfPredicate.ANCHOR_TEXT, reverse_key=True, casting_fn=self.get_resource_by_iri)
        # filter out non-entity references
        sf_reference_counts = {lex: {res: cnt for res, cnt in resources.items() if isinstance(res, DbpEntity)} for lex, resources in sf_reference_counts.items()}
        # make sure that redirects are taken into account
        for lex, resources in sf_reference_counts.items():
            for res in set(resources):
                redirect_res = self.resolve_redirect(res)
                if res != redirect_res:
                    sf_reference_counts[lex][redirect_res] += sf_reference_counts[lex][res]
                    del sf_reference_counts[lex][res]
        sf_reference_frequencies = {sub: {obj.idx: count / sum(sf_reference_counts[sub].values()) for obj, count in obj_counts.items()} for sub, obj_counts in sf_reference_counts.items()}
        return sf_reference_frequencies

    def get_wikilinks(self, res: DbpResource) -> Set[DbpResource]:
        if self.wikilinks is None:
            self.wikilinks = utils.load_or_create_cache('dbpedia_resource_wikilinks', self._init_wikilinks_cache)
        return {self.get_resource_by_idx(idx) for idx in self.wikilinks[res.idx]}

    def _init_wikilinks_cache(self) -> Dict[int, Set[int]]:
        wikilinks = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.wikilinks')], RdfPredicate.WIKILINK, casting_fn=self.get_resource_by_iri)
        wikilinks = {r.idx: {wl.idx for wl in wls} for r, wls in wikilinks.items()}
        return wikilinks

    def get_types(self, res: DbpResource) -> Set[DbpType]:
        if self.types is None:
            self.types = utils.load_or_create_cache('dbpedia_resource_types', self._init_types_cache)
        return {self.dbo.get_class_by_idx(idx) for idx in self.types[res.idx]}

    def _init_types_cache(self) -> Dict[int, Set[int]]:
        iris = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.instance_types')], RdfPredicate.TYPE)
        types = {self.get_resource_by_iri(res_iri).idx: {self.dbo.get_class_by_iri(t).idx for t in type_iris} for res_iri, type_iris in iris.items() if self.has_resource_with_iri(res_iri)}
        return types

    def get_transitive_types(self, res: DbpResource, include_root=False) -> set:
        return {tt for t in self.get_types(res) for tt in self.dbo.get_transitive_supertypes(t, include_root=include_root, include_self=True)}

    def get_entities_of_type(self, t: DbpType) -> Set[DbpEntity]:
        if self.entities_of_type is None:
            self.entities_of_type = defaultdict(set)
            for r in self.resources_by_idx.values():
                if r.is_meta or not isinstance(r, DbpEntity):
                    continue
                for it in self.dbo.get_independent_types(self.get_types(r)):
                    self.entities_of_type[it].add(r)
        return self.entities_of_type[t]

    def get_properties(self, res: DbpResource, as_tuple=False) -> Union[Dict[DbpPredicate, set], Set[Tuple[DbpPredicate, Any]]]:
        if self.properties is None:
            self.properties = utils.load_or_create_cache('dbpedia_resource_properties', self._init_property_cache)
        properties = self.properties[res.idx]
        return {(k, v) for k, vals in properties.items() for v in vals} if as_tuple else properties

    def _init_property_cache(self) -> Dict[int, Dict[DbpPredicate, set]]:
        # caution: we do not convert predicates and values to indices as we do not know whether we have literal or entity objects
        properties = defaultdict(dict)
        object_property_uris = rdf_util.create_dict_from_rdf([utils.get_data_file('files.dbpedia.mappingbased_objects')], casting_fn=self.get_resource_by_iri)
        for sub, props in object_property_uris.items():
            for pred, vals in props.items():
                pred = self.dbo.get_class_by_iri(pred)
                properties[sub.idx][pred] = vals
        literal_property_uris = rdf_util.create_dict_from_rdf([utils.get_data_file('files.dbpedia.mappingbased_literals')], casting_fn=self.get_resource_by_iri)
        for sub, props in literal_property_uris.items():
            for pred, vals in props.items():
                pred = self.dbo.get_class_by_iri(pred)
                properties[sub.idx][pred] = vals
        return dict(properties)

    def get_entity_properties(self, as_tuple=False) -> Union[Dict[DbpEntity, Dict[DbpPredicate, set]], Dict[DbpEntity, Set[Tuple[DbpPredicate, Any]]]]:
        return {e: self.get_properties(e, as_tuple) for e in self.get_entities()}

    def get_inverse_properties(self, res: DbpResource) -> Dict[DbpPredicate, Set[DbpResource]]:
        if self.inverse_properties is None:
            self.inverse_properties = defaultdict(lambda: defaultdict(set))
            for r in self.resources_by_idx.values():
                for pred, vals in self.get_properties(r):
                    if not isinstance(pred, DbpObjectPredicate):
                        continue
                    for val in vals:
                        self.inverse_properties[val][pred].add(r)
        return self.inverse_properties[res]

    def get_inverse_entity_properties(self) -> Dict[DbpEntity, Dict[DbpPredicate, set]]:
        return {e: self.get_inverse_properties(e) for e in self.get_entities()}

    def resolve_spelling_redirect(self, res: DbpResource) -> DbpResource:
        redirect_res = self.resolve_redirect(res)
        if levenshtein(res.name, redirect_res.name, 2) > 2:
            return res  # return original resource if the redirect links to a completely different resource
        return redirect_res

    @cache
    def resolve_redirect(self, res: DbpResource) -> DbpResource:
        return self.get_resource_by_idx(self._resolve_redirect_internal(res.idx))

    def _resolve_redirect_internal(self, res_idx: int, visited=None) -> int:
        if self.redirects is None:
            self.redirects = utils.load_or_create_cache('dbpedia_resource_redirects', self._init_redirect_cache)
        visited = visited or set()
        if res_idx not in self.redirects or res_idx in visited:
            return res_idx
        return self._resolve_redirect_internal(self.redirects[res_idx], visited | {res_idx})

    def _init_redirect_cache(self) -> Dict[int, int]:
        redirects = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.redirects')], RdfPredicate.REDIRECTS, casting_fn=self.get_resource_by_iri)
        return {source.idx: target.idx for source, target in redirects.items()}
