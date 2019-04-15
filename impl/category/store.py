from . import util as cat_util
import impl.util.rdf as rdf_util
import util
from collections import defaultdict
import impl.dbpedia.store as dbp_store


def get_all_cats() -> set:
    global __CATEGORIES__
    if '__CATEGORIES__' not in globals():
        initializer = lambda: set(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_TYPE))
        __CATEGORIES__ = util.load_or_create_cache('dbpedia_categories', initializer)

    return __CATEGORIES__


def get_usable_cats() -> set:
    return {cat for cat in get_all_cats() if is_usable(cat)}


def is_usable(category: str) -> bool:
    # filtering maintenance categories
    if category in get_maintenance_cats():
        return False
    # filtering other administrative categories
    indicators = ['wikipedia', 'wikiproject', 'lists', 'redirects', 'mediawiki', 'template', 'user', 'portal', 'categories', 'articles', 'pages', 'navigational', 'stubs']
    lower_category = category.lower()
    if any(indicator in lower_category for indicator in indicators):
        return False
    return True


def get_label(category: str) -> str:
    global __CATEGORY_LABELS__
    if '__CATEGORY_LABELS__' not in globals():
        __CATEGORY_LABELS__ = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_SKOS_LABEL)

    return __CATEGORY_LABELS__[category] if category in __CATEGORY_LABELS__ else cat_util.category2name(category)


def get_label_category(label: str) -> str:
    global __INVERSE_CATEGORY_LABELS__
    if '__INVERSE_CATEGORY_LABELS__' not in globals():
        labels = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_SKOS_LABEL)
        __INVERSE_CATEGORY_LABELS__ = {v: k for k, v in labels.items()}
    return __INVERSE_CATEGORY_LABELS__[label] if label in __INVERSE_CATEGORY_LABELS__ else cat_util.name2category(label)


def get_resources(category: str) -> set:
    global __CATEGORY_RESOURCES__
    if '__CATEGORY_RESOURCES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.article_categories')], rdf_util.PREDICATE_SUBJECT, reverse_key=True)
        __CATEGORY_RESOURCES__ = util.load_or_create_cache('dbpedia_category_resources', initializer)

    return __CATEGORY_RESOURCES__[category]


def get_resource_to_cats_mapping() -> dict:
    if '__RESOURCE_CATEGORIES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.article_categories')], rdf_util.PREDICATE_SUBJECT)
        global __RESOURCE_CATEGORIES__
        __RESOURCE_CATEGORIES__ = util.load_or_create_cache('dbpedia_resource_categories', initializer)

    return __RESOURCE_CATEGORIES__


def get_children(category: str) -> set:
    global __CHILDREN__
    if '__CHILDREN__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_BROADER, reverse_key=True)
        __CHILDREN__ = util.load_or_create_cache('dbpedia_category_children', initializer)

    return __CHILDREN__[category].difference({category})


def get_redirects(category: str) -> set:
    global __REDIRECTS__
    if '__REDIRECTS__' not in globals():
        __REDIRECTS__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.category_redirects')], rdf_util.PREDICATE_REDIRECTS)

    return __REDIRECTS__[category]


def get_topics(category: str) -> set:
    global __TOPICS__
    if '__TOPICS__' not in globals():
        __TOPICS__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.topical_concepts')], rdf_util.PREDICATE_SUBJECT)

    return __TOPICS__[category]


def get_maintenance_cats() -> set:
    global __MAINTENANCE_CATS__
    if '__MAINTENANCE_CATS__' not in globals():
        __MAINTENANCE_CATS__ = set(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.maintenance_categories')], rdf_util.PREDICATE_TYPE))

    return __MAINTENANCE_CATS__


def get_statistics(category: str) -> dict:
    global __CATEGORY_STATISTICS__
    if '__CATEGORY_STATISTICS__' not in globals():
        __CATEGORY_STATISTICS__ = util.load_or_create_cache('dbpedia_category_statistics', _compute_category_statistics)

    return __CATEGORY_STATISTICS__[category]


def _compute_category_statistics() -> dict:
    util.get_logger().info('Computing category statistics..')
    category_statistics = {}
    for cat in get_all_cats():
        type_counts = defaultdict(int)
        property_counts = defaultdict(int)
        property_counts_inv = defaultdict(int)
        predicate_counts = defaultdict(int)
        predicate_counts_inv = defaultdict(int)

        resources = get_resources(cat)
        for res in resources:
            resource_statistics = _compute_resource_statistics(res)
            for t in resource_statistics['types']:
                type_counts[t] += 1
            for prop in resource_statistics['properties']:
                property_counts[prop] += 1
            for prop in resource_statistics['properties_inv']:
                property_counts_inv[prop] += 1
            for pred in resource_statistics['predicates']:
                predicate_counts[pred] += 1
            for pred in resource_statistics['predicates_inv']:
                predicate_counts_inv[pred] += 1
        category_statistics[cat] = {
            'type_counts': type_counts,
            'type_frequencies': defaultdict(float, {t: t_count / len(resources) for t, t_count in type_counts.items()}),
            'property_counts': property_counts,
            'property_counts_inv': property_counts_inv,
            'property_frequencies': defaultdict(float, {prop: p_count / len(resources) for prop, p_count in property_counts.items()}),
            'property_frequencies_inv': defaultdict(float, {prop: p_count / len(resources) for prop, p_count in property_counts_inv.items()}),
            'predicate_counts': predicate_counts,
            'predicate_counts_inv': predicate_counts_inv
        }
    return category_statistics


def _compute_resource_statistics(dbp_resource: str) -> dict:
    global __RESOURCE_STATISTICS__
    if '__RESOURCE_STATISTICS__' not in globals():
        __RESOURCE_STATISTICS__ = {}
    if dbp_resource not in __RESOURCE_STATISTICS__:
        __RESOURCE_STATISTICS__[dbp_resource] = {
            'types': dbp_store.get_transitive_types(dbp_resource),
            'properties': {(pred, val) for pred, values in dbp_store.get_properties(dbp_resource).items() for val in values},
            'properties_inv': {(pred, val) for pred, values in dbp_store.get_inverse_properties(dbp_resource).items() for val in values},
            'predicates': {pred for pred in dbp_store.get_properties(dbp_resource)},
            'predicates_inv': {pred for pred in dbp_store.get_inverse_properties(dbp_resource)},
        }
    return __RESOURCE_STATISTICS__[dbp_resource]
