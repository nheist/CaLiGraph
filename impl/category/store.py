"""Functionality to retrieve everything related to categories."""

from . import util as cat_util
import impl.util.rdf as rdf_util
import impl.dbpedia.store as dbp_store
import impl.wikipedia as wikipedia
import util
from collections import defaultdict


def get_categories() -> set:
    """Return all categories."""
    global __CATEGORIES__
    if '__CATEGORIES__' not in globals():
        initializer = lambda: set(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.category_skos')], rdf_util.PREDICATE_TYPE))
        __CATEGORIES__ = util.load_or_create_cache('dbpedia_categories', initializer)

    return __CATEGORIES__


def get_usable_categories() -> set:
    """Return only usable categories, i.e. categories that are not used for maintenance and organisational purposes.
    See https://en.wikipedia.org/wiki/Category:Wikipedia_categories for an overview of maintenance categories.
    """
    global __USABLE_CATEGORIES__
    if '__USABLE_CATEGORIES__' not in globals():
        __USABLE_CATEGORIES__ = get_categories()
        maintenance_parent_categories = {
            'Disambiguation categories', 'Hidden categories', 'Maintenance categories', 'Namespace-specific categories',
            'All redirect categories', 'Wikipedia soft redirected categories', 'Tracking categories'
        }
        for mpc in maintenance_parent_categories:
            mpc_child_closure = {mpc} | get_transitive_children(mpc)
            __USABLE_CATEGORIES__ = __USABLE_CATEGORIES__.difference(mpc_child_closure)

        maintenance_category_indicators = {
            'wikipedia', 'wikipedians', 'wikimedia', 'wikiproject', 'lists', 'redirects', 'mediawiki', 'template',
            'templates', 'user', 'portal', 'categories', 'articles', 'pages', 'navigational', 'stubs'
        }
        maintenance_categories = set()
        for c in __USABLE_CATEGORIES__:
            category_tokens = {t.lower() for t in cat_util.remove_category_prefix(c).split('_')}
            if category_tokens.intersection(maintenance_category_indicators):
                maintenance_categories.add(c)
        __USABLE_CATEGORIES__ = __USABLE_CATEGORIES__.difference(maintenance_categories)

    return __USABLE_CATEGORIES__


def get_label(category: str) -> str:
    """Return the label for the given category."""
    global __CATEGORY_LABELS__
    if '__CATEGORY_LABELS__' not in globals():
        __CATEGORY_LABELS__ = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.category_skos')], rdf_util.PREDICATE_PREFLABEL)

    return __CATEGORY_LABELS__[category] if category in __CATEGORY_LABELS__ else cat_util.category2name(category)


def get_label_category(label: str) -> str:
    """Return the category that fits the given label best."""
    global __INVERSE_CATEGORY_LABELS__
    if '__INVERSE_CATEGORY_LABELS__' not in globals():
        labels = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.category_skos')], rdf_util.PREDICATE_PREFLABEL)
        __INVERSE_CATEGORY_LABELS__ = {v: k for k, v in labels.items()}
    return __INVERSE_CATEGORY_LABELS__[label] if label in __INVERSE_CATEGORY_LABELS__ else cat_util.name2category(label)


def get_resources(category: str) -> set:
    """Return all resources of the given category."""
    global __CATEGORY_RESOURCES__
    if '__CATEGORY_RESOURCES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.category_articles')], rdf_util.PREDICATE_SUBJECT, reverse_key=True)
        __CATEGORY_RESOURCES__ = util.load_or_create_cache('dbpedia_category_resources', initializer)

    return __CATEGORY_RESOURCES__[category]


def get_resource_categories(dbp_resource: str) -> set:
    """Return all categories the given resource is contained in."""
    global __RESOURCE_CATEGORIES__
    if '__RESOURCE_CATEGORIES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.category_articles')], rdf_util.PREDICATE_SUBJECT)
        __RESOURCE_CATEGORIES__ = util.load_or_create_cache('dbpedia_resource_categories', initializer)

    return __RESOURCE_CATEGORIES__[dbp_resource]


def get_parents(category: str) -> set:
    """Return all direct supercategories for the given category."""
    global __PARENTS__
    if '__PARENTS__' not in globals():
        # use parent category relationships extracted from dbpedia extraction framework
        __PARENTS__ = defaultdict(set, rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.category_skos')], rdf_util.PREDICATE_BROADER))
        # combine them with parent categories extracted directly from wikimarkup (including parent categories expressed in templates)
        for child, parents in wikipedia.extract_parent_categories().items():
            __PARENTS__[child].update(parents)
        __PARENTS__ = defaultdict(set, {child: parents.difference({child}) for child, parents in __PARENTS__.items()})
    return __PARENTS__[category]


def get_children(category: str) -> set:
    """Return all direct subcategories for the given category."""
    global __CHILDREN__, __PARENTS__
    if '__CHILDREN__' not in globals():
        get_parents('')  # make sure that __PARENTS__ is initialized
        __CHILDREN__ = defaultdict(set)
        for child, parents in __PARENTS__.items():
            for parent in parents:
                __CHILDREN__[parent].add(child)
    return __CHILDREN__[category]


def get_transitive_children(category: str) -> set:
    """Return all (including transitive) subcategories for the given category."""
    children = get_children(category)
    return children | {tc for c in children for tc in get_transitive_children(c)}


def get_topics(category: str) -> set:
    """Return the topics for the given category."""
    global __TOPICS__
    if '__TOPICS__' not in globals():
        __TOPICS__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.topical_concepts')], rdf_util.PREDICATE_SUBJECT)

    return __TOPICS__[category]


def get_topic_categories(dbp_resource: str) -> set:
    """Return all categories the given resource is a topic of."""
    global __TOPIC_CATEGORIES__
    if '__TOPIC_CATEGORIES__' not in globals():
        __TOPIC_CATEGORIES__ = defaultdict(set)
        for cat in get_categories():
            for topic in get_topics(cat):
                __TOPIC_CATEGORIES__[topic].add(cat)
    return __TOPIC_CATEGORIES__[dbp_resource]


def get_maintenance_categories() -> set:
    """Return all categories that are used solely for maintenance purposes in Wikipedia."""
    global __MAINTENANCE_CATS__
    if '__MAINTENANCE_CATS__' not in globals():
        __MAINTENANCE_CATS__ = set(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.maintenance_categories')], rdf_util.PREDICATE_TYPE))

    return __MAINTENANCE_CATS__


def get_statistics(category: str) -> dict:
    """Return information about the amounts/frequencies of types and properties of a category's resources."""
    global __CATEGORY_STATISTICS__
    if '__CATEGORY_STATISTICS__' not in globals():
        __CATEGORY_STATISTICS__ = util.load_or_create_cache('dbpedia_category_statistics', _compute_category_statistics)
    return __CATEGORY_STATISTICS__[category]


def _compute_category_statistics() -> dict:
    category_statistics = {}
    for cat in get_categories():
        type_counts = defaultdict(int)
        property_counts = defaultdict(int)

        resources = get_resources(cat)
        for res in resources:
            resource_statistics = dbp_store.get_statistics(res)
            for t in resource_statistics['types']:
                type_counts[t] += 1
            for prop in resource_statistics['properties']:
                property_counts[prop] += 1
        category_statistics[cat] = {
            'type_counts': type_counts,
            'type_frequencies': defaultdict(float, {t: t_count / len(resources) for t, t_count in type_counts.items()}),
            'property_counts': property_counts,
            'property_frequencies': defaultdict(float, {prop: p_count / len(resources) for prop, p_count in property_counts.items()}),
        }
    return category_statistics
