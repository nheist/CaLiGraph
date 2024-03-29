from typing import Dict, Set, Optional, Iterable, Tuple
from collections import defaultdict, Counter
import re
from impl.util.singleton import Singleton
import impl.dbpedia.util as dbp_util
import impl.util.rdf as rdf_util
from impl.util.rdf import RdfPredicate, RdfResource, Namespace
from impl.dbpedia.resource import DbpResourceStore, DbpResource, DbpEntity
from impl import wikipedia
import utils


class DbpCategory(RdfResource):
    def get_resources(self) -> Set[DbpResource]:
        return self._get_store().get_resources(self)

    def get_entities(self) -> Set[DbpEntity]:
        return {r for r in self.get_resources() if isinstance(r, DbpEntity) and not r.is_meta}

    def get_statistics(self) -> dict:
        return self._get_store().get_statistics(self)

    @classmethod
    def get_namespace(cls) -> str:
        return Namespace.DBP_RESOURCE.value

    @classmethod
    def _get_prefixes(cls) -> set:
        return {Namespace.PREFIX_CATEGORY.value}

    @classmethod
    def _get_store(cls):
        return DbpCategoryStore.instance()


class DbpListCategory(DbpCategory):
    @classmethod
    def _get_prefixes(cls) -> set:
        return {Namespace.PREFIX_LISTS.value, Namespace.PREFIX_LISTCATEGORY.value}


class DbpCategoryNotExistingException(KeyError):
    pass


@Singleton
class DbpCategoryStore:
    def __init__(self):
        self.dbr = DbpResourceStore.instance()

        all_categories = utils.load_or_create_cache('dbpedia_categories', self._init_category_cache)
        self.categories_by_idx = {c.idx: c for c in all_categories}
        self.categories_by_name = {c.name: c for c in all_categories}
        self.children = None
        self.parents = None
        self.labels = None
        self.resources = None
        self.resource_categories = None
        self.topics = None
        self.categories_by_topic = None
        self.statistics = {}

    def _init_category_cache(self) -> Set[DbpCategory]:
        # gather categories
        category_uris = rdf_util.create_set_from_rdf([utils.get_data_file('files.dbpedia.category_skos')], RdfPredicate.TYPE, None)
        category_names = {dbp_util.resource_iri2name(uri) for uri in category_uris}
        category_names = category_names.difference({utils.get_config('category.root_category')})  # root category will be treated separately
        # gather category hierarchy
        category_children_iris = self._load_category_children_iris()
        category_children = [(dbp_util.resource_iri2name(p), dbp_util.resource_iri2name(c)) for p, c in category_children_iris]
        # identify meta categories
        meta_parent_category_titles = {'Hidden_categories', 'Tracking_categories', 'Disambiguation_categories',
                                       'Non-empty_disambiguation_categories', 'All_redirect_categories',
                                       'Wikipedia_soft_redirected_categories', 'Category_redirects_with_possibilities',
                                       'Wikipedia_non-empty_soft_redirected_categories'}
        meta_parent_categories = {Namespace.PREFIX_CATEGORY.value + cat for cat in meta_parent_category_titles}
        meta_categories = meta_parent_categories | {c for p, c in category_children if p in meta_parent_categories}
        # identify any remaining invalid categories (maintenance categories etc) using indicator tokens
        ignored_category_endings = ('files', 'images', 'lists', 'articles', 'stubs', 'pages', 'categories')
        maintenance_category_indicators = {'wikipedia', 'wikipedians', 'wikimedia', 'wikiproject', 'redirects',
                                           'mediawiki', 'template', 'templates', 'user', 'portal', 'navigational'}
        meta_categories.update({c for c in category_names if c.lower().endswith(ignored_category_endings) or set(re.split('[_:]', c.lower())).intersection(maintenance_category_indicators)})

        # build actual category classes
        categories = {DbpCategory(0, utils.get_config('category.root_category'), False)}
        for idx, c in enumerate(category_names, start=1):
            if c.startswith(Namespace.PREFIX_LISTCATEGORY.value):
                categories.add(DbpListCategory(idx, c, c in meta_categories))
            else:
                categories.add(DbpCategory(idx, c, c in meta_categories))
        return categories

    @classmethod
    def _load_category_children_iris(cls) -> Set[Tuple[str, str]]:
        skos_category_parent_iris = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.category_skos')], RdfPredicate.BROADER)
        category_children_iris = {(p, c) for c, parents in skos_category_parent_iris.items() for p in parents}
        wiki_category_parent_uris = wikipedia.extract_parent_categories()
        category_children_iris.update({(p, c) for c, parents in wiki_category_parent_uris.items() for p in parents})
        category_children_iris = {(p, c) for p, c in category_children_iris if p != c}
        return category_children_iris

    def has_category_with_idx(self, idx: int) -> bool:
        return idx in self.categories_by_idx

    def get_category_by_idx(self, idx: int) -> DbpCategory:
        if not self.has_category_with_idx(idx):
            raise DbpCategoryNotExistingException(f'Could not find category for index: {idx}')
        return self.categories_by_idx[idx]

    def has_category_with_name(self, name: str) -> bool:
        return name in self.categories_by_name

    def get_category_by_name(self, name: str) -> DbpCategory:
        if not self.has_category_with_name(name):
            raise DbpCategoryNotExistingException(f'Could not find category for name: {name}')
        return self.categories_by_name[name]

    def has_category_with_iri(self, iri: str) -> bool:
        return self.has_category_with_name(dbp_util.resource_iri2name(iri))

    def get_category_by_iri(self, iri: str) -> DbpCategory:
        return self.categories_by_name[dbp_util.resource_iri2name(iri)]

    def get_categories(self, include_meta=False, include_listcategories=False) -> Set[DbpCategory]:
        return self._filter_categories(self.categories_by_idx.values(), include_meta, include_listcategories)

    def get_listcategories(self) -> Set[DbpListCategory]:
        return {c for c in self.categories_by_idx.values() if isinstance(c, DbpListCategory)}

    def get_category_root(self):
        return self.get_category_by_idx(0)  # root is initialized as 0

    def get_children(self, cat: DbpCategory, include_meta=False, include_listcategories=False) -> Set[DbpCategory]:
        if self.children is None:
            category_children = utils.load_or_create_cache('dbpedia_category_children', self._init_category_children_cache)
            self.children = defaultdict(set, category_children)
        children = {self.get_category_by_idx(idx) for idx in self.children[cat.idx]}
        return self._filter_categories(children, include_meta, include_listcategories)

    def _init_category_children_cache(self) -> Dict[int, Set[int]]:
        category_children = defaultdict(set)
        for p_iri, c_iri in self._load_category_children_iris():
            if not self.has_category_with_iri(p_iri) or not self.has_category_with_iri(c_iri):
                continue
            p_cat = self.get_category_by_iri(p_iri)
            c_cat = self.get_category_by_iri(c_iri)
            # HOTFIX: Make sure that Person and Ethnic_group are not directly connected
            if p_cat.name == 'Category:Ethnic_groups' and c_cat.name == 'Category:People_by_ethnicity':
                continue
            category_children[p_cat.idx].add(c_cat.idx)
        return category_children

    def get_parents(self, cat: DbpCategory, include_meta=False, include_listcategories=False) -> Set[DbpCategory]:
        if self.parents is None:
            self.parents = defaultdict(set)
            for cat in self.get_categories(include_meta=True, include_listcategories=True):
                for child in self.get_children(cat, include_meta=True, include_listcategories=True):
                    self.parents[child.idx].add(cat.idx)
        parents = {self.get_category_by_idx(idx) for idx in self.parents[cat.idx]}
        return self._filter_categories(parents, include_meta, include_listcategories)

    def _filter_categories(self, categories: Iterable[DbpCategory], include_meta: bool, include_listcategories: bool) -> Set[DbpCategory]:
        if not include_listcategories:
            categories = {c for c in categories if not isinstance(c, DbpListCategory)}
        if not include_meta:
            categories = {c for c in categories if not c.is_meta}
        return categories

    def get_label(self, cat: DbpCategory) -> Optional[str]:
        if self.labels is None:
            labels = utils.load_or_create_cache('dbpedia_category_labels', self._init_label_cache)
            self.labels = defaultdict(lambda: None, labels)
        return self.labels[cat.idx]

    def _init_label_cache(self) -> Dict[int, str]:
        labels = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.category_skos')], RdfPredicate.PREFLABEL, casting_fn=self.get_category_by_iri)
        labels = {cat.idx: label for cat, label in labels.items()}
        return labels

    def get_resources(self, cat: DbpCategory) -> Set[DbpResource]:
        if self.resources is None:
            resources = utils.load_or_create_cache('dbpedia_category_resources', self._init_category_resource_cache)
            self.resources = defaultdict(set, resources)
        return {self.dbr.get_resource_by_idx(idx) for idx in self.resources[cat.idx]}

    def _init_category_resource_cache(self) -> Dict[int, Set[int]]:
        category_resource_iris = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.category_articles')], RdfPredicate.SUBJECT, reverse_key=True)
        category_resources = {}
        for cat_iri, resource_iris in category_resource_iris.items():
            if not self.has_category_with_iri(cat_iri):
                continue
            category_resources[self.get_category_by_iri(cat_iri).idx] = {self.dbr.get_resource_by_iri(res_iri).idx for res_iri in resource_iris}
        return category_resources

    def get_categories_for_resource(self, res: DbpResource) -> Set[DbpCategory]:
        if self.resource_categories is None:
            self.resource_categories = defaultdict(set)
            for c in self.get_categories(include_meta=True, include_listcategories=True):
                for r in self.get_resources(c):
                    self.resource_categories[r].add(c)
        return self.resource_categories[res]

    def get_topics(self, cat: DbpCategory) -> Set[DbpResource]:
        if self.topics is None:
            self.topics = defaultdict(set)
            category_topic_iris = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.topical_concepts')], RdfPredicate.SUBJECT)
            for cat_iri, topic_iris in category_topic_iris.items():
                if not self.has_category_with_iri(cat_iri):
                    continue
                self.topics[self.get_category_by_iri(cat_iri)] = {self.dbr.get_resource_by_iri(topic_iri) for topic_iri in topic_iris if self.dbr.has_resource_with_iri(topic_iri)}
        return self.topics[cat]

    def get_categories_for_topic(self, res: DbpResource, include_meta=False, include_listcategories=False) -> Set[DbpCategory]:
        if self.categories_by_topic is None:
            self.categories_by_topic = defaultdict(set)
            for c in self.get_categories(include_meta=True, include_listcategories=True):
                for t in self.get_topics(c):
                    self.categories_by_topic[t].add(c)
        categories = self.categories_by_topic[res]
        return self._filter_categories(categories, include_meta, include_listcategories)

    def get_statistics(self, cat: DbpCategory) -> dict:
        if cat not in self.statistics:
            type_counts = Counter()
            prop_counts = Counter()

            resources = self.get_resources(cat)
            for res in resources:
                type_counts.update(res.get_transitive_types())
                prop_counts.update(res.get_properties(as_tuple=True))
            self.statistics[cat] = {
                'type_counts': type_counts,
                'type_frequencies': defaultdict(float, {t: t_count / len(resources) for t, t_count in type_counts.items()}),
                'property_counts': prop_counts,
                'property_frequencies': defaultdict(float, {prop: p_count / len(resources) for prop, p_count in prop_counts.items()}),
            }
        return self.statistics[cat]
