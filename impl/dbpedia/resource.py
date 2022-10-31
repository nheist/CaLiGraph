from typing import Union, Dict, Optional, Set, Any, Tuple, List
from collections import defaultdict, Counter
from impl.util.singleton import Singleton
import impl.util.rdf as rdf_util
from impl.util.rdf import Namespace, RdfPredicate, RdfResource
from impl.util.nlp import EntityTypeLabel, TYPE_TO_LABEL_MAPPING
import impl.dbpedia.util as dbp_util
from impl.dbpedia.ontology import DbpType, DbpPredicate, DbpObjectPredicate, DbpOntologyStore
from polyleven import levenshtein
import utils
import bz2
import csv


class DbpResource(RdfResource):
    def get_surface_forms(self) -> Set[str]:
        return self._get_store().get_surface_forms(self)

    def get_types(self) -> Set[DbpType]:
        return self._get_store().get_types(self)

    def get_independent_types(self) -> Set[DbpType]:
        return self._get_store().get_independent_types(self)

    def get_transitive_types(self, include_root=False) -> Set[DbpType]:
        return self._get_store().get_transitive_types(self, include_root=include_root)

    def get_type_label(self) -> EntityTypeLabel:
        return self._get_store().get_type_label(self.idx)

    def get_properties(self, as_tuple=False) -> Union[Dict[DbpPredicate, set], Set[Tuple[DbpPredicate, Any]]]:
        return self._get_store().get_properties(self, as_tuple)

    @classmethod
    def get_namespace(cls) -> str:
        return Namespace.DBP_RESOURCE.value

    @classmethod
    def _get_store(cls):
        return DbpResourceStore.instance()


class DbpEntity(DbpResource):
    def get_abstract(self) -> Optional[str]:
        return self._get_store().get_abstract(self)

    def get_embedding_vector(self) -> Optional[List[float]]:
        return self._get_store().get_embedding_vectors()[self.idx]


class DbpListpage(DbpResource):
    def get_label(self) -> str:
        label = super().get_label()
        return label[4:] if label.startswith('the ') else label

    @classmethod
    def _get_prefixes(cls) -> set:
        return {Namespace.PREFIX_LIST.value}


class DbpFile(DbpResource):
    @classmethod
    def _get_prefixes(cls) -> set:
        return {Namespace.PREFIX_FILE.value, Namespace.PREFIX_IMAGE.value}


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
        self.abstracts = None
        self.page_ids = None
        self.wikidata_links = None

        self.types = None
        self.entities_of_type = None
        self.type_labels = None
        self.wikilinks = None
        self.properties = None
        self.inverse_properties = None
        self.redirects = None
        self.embedding_vectors = None

    def _init_resource_cache(self) -> List[DbpResource]:
        # find all resources that have at least a label or a type
        resources_with_label = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.labels')], RdfPredicate.LABEL, None)
        resources_with_type = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.instance_types')], RdfPredicate.TYPE, None)
        all_resource_uris = {dbp_util.get_canonical_iri(uri) for uri in resources_with_label | resources_with_type}  # make all uris canonical
        all_resource_uris = {uri for uri in all_resource_uris if dbp_util.is_resource_iri(uri) and not dbp_util.is_category_iri(uri)}  # filter out invalid uris
        # find resources that are redirects or disambiguations
        meta_resources = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.redirects')], RdfPredicate.REDIRECTS, None)
        meta_resources.update(rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.disambiguations')], RdfPredicate.DISAMBIGUATES, None))

        all_resources = []
        for idx, res_uri in enumerate(all_resource_uris):
            res_name = dbp_util.resource_iri2name(res_uri)
            is_meta = res_uri in meta_resources
            if dbp_util.is_listpage_iri(res_uri):
                all_resources.append(DbpListpage(idx, res_name, is_meta))
            elif dbp_util.is_file_iri(res_uri):
                all_resources.append(DbpFile(idx, res_name, is_meta))
            else:
                all_resources.append(DbpEntity(idx, res_name, is_meta))
        return all_resources

    def has_resource_with_idx(self, idx: int) -> bool:
        return idx in self.resources_by_idx

    def get_resource_by_idx(self, idx: int) -> DbpResource:
        if not self.has_resource_with_idx(idx):
            raise DbpResourceNotExistingException(f'Could not find resource for index: {idx}')
        return self.resources_by_idx[idx]

    def get_highest_resource_idx(self) -> int:
        return max(self.resources_by_idx)

    def has_resource_with_name(self, name: str) -> bool:
        return name in self.resources_by_name

    def get_resource_by_name(self, name: str) -> DbpResource:
        if not self.has_resource_with_name(name):
            raise DbpResourceNotExistingException(f'Could not find resource for name: {name}')
        return self.resources_by_name[name]

    def has_resource_with_iri(self, iri: str) -> bool:
        return self.has_resource_with_name(dbp_util.resource_iri2name(iri))

    def get_resource_by_iri(self, iri: str) -> DbpResource:
        if not self.has_resource_with_iri(iri):
            raise DbpResourceNotExistingException(f'Could not find resource for iri: {iri}')
        return self.get_resource_by_name(dbp_util.resource_iri2name(iri))

    def get_resources(self) -> Set[DbpResource]:
        return set(self.resources_by_idx.values())

    def get_entities(self, include_meta=False) -> Set[DbpEntity]:
        entities = {r for r in self.resources_by_idx.values() if isinstance(r, DbpEntity)}
        entities = entities if include_meta else {e for e in entities if not e.is_meta}
        return entities

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
            self.surface_forms = defaultdict(set, utils.load_or_create_cache('dbpedia_resource_surface_forms', self._init_surface_form_cache))
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
        sf_reference_counts = {lex: Counter({res: cnt for res, cnt in resources.items() if isinstance(res, DbpEntity)}) for lex, resources in sf_reference_counts.items()}
        sf_reference_counts = defaultdict(Counter, sf_reference_counts)
        # make sure that redirects are taken into account
        for lex, resources in sf_reference_counts.items():
            for res in set(resources):
                redirect_res = self.resolve_redirect(res)
                if res != redirect_res:
                    sf_reference_counts[lex][redirect_res] += sf_reference_counts[lex][res]
                    del sf_reference_counts[lex][res]
        sf_reference_frequencies = {sub: {obj.idx: count / obj_counts.total() for obj, count in obj_counts.items()} for sub, obj_counts in sf_reference_counts.items()}
        return sf_reference_frequencies

    def get_wikilinks(self, res: DbpResource) -> Set[DbpResource]:
        if self.wikilinks is None:
            self.wikilinks = defaultdict(set, utils.load_or_create_cache('dbpedia_resource_wikilinks', self._init_wikilinks_cache))
        return {self.get_resource_by_idx(idx) for idx in self.wikilinks[res.idx]}

    def _init_wikilinks_cache(self) -> Dict[int, Set[int]]:
        wikilinks = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.wikilinks')], RdfPredicate.WIKILINK, casting_fn=self.get_resource_by_iri)
        return {r.idx: {wl.idx for wl in wls} for r, wls in wikilinks.items()}

    def get_abstract(self, res: DbpResource) -> Optional[str]:
        if self.abstracts is None:
            self.abstracts = defaultdict(lambda: None, utils.load_or_create_cache('dbpedia_resource_abstracts', self._init_abstracts_cache))
        return self.abstracts[res.idx]

    def _init_abstracts_cache(self) -> Dict[int, Optional[str]]:
        abstracts = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.short_abstracts')], RdfPredicate.COMMENT, casting_fn=self.get_resource_by_iri)
        return {r.idx: abstract for r, abstract in abstracts.items()}

    def get_types(self, res: DbpResource) -> Set[DbpType]:
        if self.types is None:
            self.types = defaultdict(set, utils.load_or_create_cache('dbpedia_resource_types', self._init_types_cache))
        return {self.dbo.get_class_by_idx(idx) for idx in self.types[res.idx]}

    def _init_types_cache(self) -> Dict[int, Set[int]]:
        iris = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.instance_types')], RdfPredicate.TYPE)
        types = {self.get_resource_by_iri(res_iri).idx: {self.dbo.get_class_by_iri(t).idx for t in type_iris} for res_iri, type_iris in iris.items() if self.has_resource_with_iri(res_iri)}
        return types

    def get_transitive_types(self, res: DbpResource, include_root=False) -> Set[DbpType]:
        return {tt for t in res.get_types() for tt in self.dbo.get_transitive_supertypes(t, include_root=include_root, include_self=True)}

    def get_independent_types(self, res: DbpResource) -> Set[DbpType]:
        return self.dbo.get_independent_types(res.get_types())

    def get_entities_of_type(self, t: DbpType) -> Set[DbpEntity]:
        if self.entities_of_type is None:
            self.entities_of_type = defaultdict(set)
            for r in self.resources_by_idx.values():
                if r.is_meta or not isinstance(r, DbpEntity):
                    continue
                for it in r.get_independent_types():
                    self.entities_of_type[it].add(r)
        return self.entities_of_type[t]

    def get_type_label(self, res_idx: int) -> EntityTypeLabel:
        if self.type_labels is None:
            self.type_labels = defaultdict(lambda: EntityTypeLabel.OTHER, utils.load_or_create_cache('dbpedia_resource_typelabels', self._init_type_label_cache))
        return self.type_labels[res_idx]

    def _init_type_label_cache(self) -> Dict[int, EntityTypeLabel]:
        type_labels = {}
        for ent in self.get_entities():
            for t in ent.get_transitive_types():
                if t.name in TYPE_TO_LABEL_MAPPING:
                    type_labels[ent.idx] = TYPE_TO_LABEL_MAPPING[t.name]
        return type_labels

    def get_properties(self, res: DbpResource, as_tuple=False) -> Union[Dict[DbpPredicate, set], Set[Tuple[DbpPredicate, Any]]]:
        if self.properties is None:
            self.properties = defaultdict(dict, utils.load_or_create_cache('dbpedia_resource_properties', self._init_property_cache))
        properties = self.properties[res.idx]
        return {(k, v) for k, vals in properties.items() for v in vals} if as_tuple else properties

    def _init_property_cache(self) -> Dict[int, Dict[DbpPredicate, set]]:
        # caution: we do not convert predicates and values to indices as we do not know whether we have literal or entity objects
        properties = defaultdict(dict)
        object_property_uris = rdf_util.create_dict_from_rdf([utils.get_data_file('files.dbpedia.mappingbased_objects')], casting_fn=self.get_resource_by_iri)
        for sub, props in object_property_uris.items():
            for pred, vals in props.items():
                if not self.dbo.has_class_with_iri(pred):
                    continue
                pred = self.dbo.get_class_by_iri(pred)
                properties[sub.idx][pred] = vals
        literal_property_uris = rdf_util.create_dict_from_rdf([utils.get_data_file('files.dbpedia.mappingbased_literals')], casting_fn=self.get_resource_by_iri)
        for sub, props in literal_property_uris.items():
            for pred, vals in props.items():
                if not self.dbo.has_class_with_iri(pred):
                    continue
                pred = self.dbo.get_class_by_iri(pred)
                properties[sub.idx][pred] = vals
        return dict(properties)

    def get_entity_properties(self, include_meta=False, as_tuple=False) -> Union[Dict[DbpEntity, Dict[DbpPredicate, set]], Dict[DbpEntity, Set[Tuple[DbpPredicate, Any]]]]:
        return {e: self.get_properties(e, as_tuple) for e in self.get_entities(include_meta=include_meta)}

    def get_inverse_properties(self, res: DbpResource) -> Dict[DbpPredicate, Set[DbpResource]]:
        if self.inverse_properties is None:
            self.inverse_properties = defaultdict(lambda: defaultdict(set))
            for r in self.resources_by_idx.values():
                for pred, vals in self.get_properties(r).items():
                    if not isinstance(pred, DbpObjectPredicate):
                        continue
                    for val in vals:
                        self.inverse_properties[val][pred].add(r)
        return self.inverse_properties[res]

    def get_inverse_entity_properties(self, include_meta=False) -> Dict[DbpEntity, Dict[DbpPredicate, set]]:
        return {e: self.get_inverse_properties(e) for e in self.get_entities(include_meta=include_meta)}

    def resolve_spelling_redirect(self, res: DbpResource) -> DbpResource:
        if not res.is_meta:
            return res
        redirect_res = self.resolve_redirect(res)
        if levenshtein(res.name, redirect_res.name, 2) > 2:
            return res  # return original resource if the redirect links to a completely different resource
        return redirect_res

    def resolve_redirect(self, res: DbpResource) -> DbpResource:
        if self.redirects is None:
            self.redirects = utils.load_or_create_cache('dbpedia_resource_redirects', self._init_redirect_cache)
        return self.get_resource_by_idx(self.redirects[res.idx]) if res.idx in self.redirects else res

    def _init_redirect_cache(self) -> Dict[int, int]:
        redirects = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.redirects')], RdfPredicate.REDIRECTS, casting_fn=self.get_resource_by_iri)
        return {source.idx: target.idx for source, target in redirects.items()}

    def get_resource_by_page_id(self, page_id: int) -> Optional[DbpResource]:
        if self.page_ids is None:
            page_id_to_res = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.page_ids')], RdfPredicate.WIKIID, reverse_key=True, casting_fn=self.get_resource_by_iri)
            self.page_ids = defaultdict(lambda: None, {int(page_id): res for res, page_id in page_id_to_res.items()})
        return self.page_ids[page_id]

    def get_resource_by_wikidata_id(self, wikidata_id: str) -> Optional[DbpResource]:
        if self.wikidata_links is None:
            wikidata_to_res = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.wikidata_links')], RdfPredicate.SAME_AS, casting_fn=self.get_resource_by_iri)
            self.wikidata_links = {wd_url[wd_url.rindex('/')+1:]: res for wd_url, resources in wikidata_to_res.items() for res in resources if isinstance(res, DbpResource)}
        return self.wikidata_links[wikidata_id]

    def get_embedding_vectors(self) -> Dict[int, List[float]]:
        if self.embedding_vectors is None:
            self.embedding_vectors = defaultdict(lambda: None, utils.load_or_create_cache('dbpedia_resource_embeddings', self._init_embedding_cache))
        return self.embedding_vectors

    def _init_embedding_cache(self) -> Dict[int, List[float]]:
        embedding_vectors = {}
        with bz2.open(utils.get_data_file('files.dbpedia.embedding_vectors'), mode='rt', newline='') as f:
            for row in csv.reader(f, delimiter=' '):
                res_iri = rdf_util.uri2iri(row[0])
                if not self.has_resource_with_iri(res_iri):
                    continue
                # discarding last parsed value as it is an empty string
                embedding_vectors[self.get_resource_by_iri(res_iri).idx] = [float(v) for v in row[1:-1]]
        return embedding_vectors
