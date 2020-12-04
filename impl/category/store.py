"""Functionality to retrieve everything related to categories."""

import impl.category.util as cat_util
import impl.util.rdf as rdf_util
import impl.dbpedia.store as dbp_store
from impl import wikipedia
import utils
from collections import defaultdict
import networkx as nx


def get_categories() -> set:
    """Return all categories that are not hidden or used as any kind of organisational category."""
    return set(_get_category_graph())


def get_parents(category: str) -> set:
    """Return all direct supercategories for the given category."""
    category_graph = _get_category_graph()
    return set(category_graph.predecessors(category)) if category in category_graph else set()


def get_children(category: str) -> set:
    """Return all direct subcategories for the given category."""
    category_graph = _get_category_graph()
    return set(category_graph.successors(category)) if category in category_graph else set()


def _get_category_graph() -> nx.DiGraph:
    global __CATEGORY_GRAPH__
    if '__CATEGORY_GRAPH__' not in globals():
        __CATEGORY_GRAPH__ = utils.load_or_create_cache('dbpedia_categories', _create_category_graph)
    return __CATEGORY_GRAPH__


def _create_category_graph() -> nx.DiGraph:
    wiki_category_edges = [(p, c) for c, ps in wikipedia.extract_parent_categories().items() for p in ps if p != c]
    graph = nx.DiGraph(incoming_graph_data=wiki_category_edges)
    # identify hidden/tracking categories
    invalid_categories = set(graph.successors(cat_util.name2category('Hidden categories')))
    invalid_categories.update(graph.successors(cat_util.name2category('Tracking categories')))
    # identify disambiguation and redirect categories
    invalid_categories.update(nx.descendants(graph, cat_util.name2category('Disambiguation categories')))
    invalid_categories.update(nx.descendants(graph, cat_util.name2category('All redirect categories')))
    # add nodes/edges from skos-categories file (this is done on purpose AFTER identification of disambiguations/redirects to avoid false edges when using descendants)
    skos_nodes = list(rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.category_skos')], rdf_util.PREDICATE_TYPE))
    graph.add_nodes_from(skos_nodes)
    skos_edges = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.category_skos')], rdf_util.PREDICATE_BROADER)
    graph.add_edges_from([(p, c) for c, parents in skos_edges.items() for p in parents if p != c])
    # identify any remaining maintenance categories
    maintenance_category_indicators = {
        'wikipedia', 'wikipedians', 'wikimedia', 'wikiproject', 'lists', 'redirects', 'mediawiki', 'template',
        'templates', 'user', 'portal', 'categories', 'articles', 'pages', 'navigational', 'stubs'
    }
    for cat in graph:
        cat_tokens = {t.lower() for t in cat_util.remove_category_prefix(cat).split('_')}
        if cat_tokens.intersection(maintenance_category_indicators):
            invalid_categories.add(cat)
    graph.remove_nodes_from(invalid_categories)
    return graph


def get_label(category: str) -> str:
    """Return the label for the given category."""
    global __CATEGORY_LABELS__
    if '__CATEGORY_LABELS__' not in globals():
        __CATEGORY_LABELS__ = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.category_skos')], rdf_util.PREDICATE_PREFLABEL)

    return __CATEGORY_LABELS__[category] if category in __CATEGORY_LABELS__ else cat_util.category2name(category)


def get_label_category(label: str) -> str:
    """Return the category that fits the given label best."""
    global __INVERSE_CATEGORY_LABELS__
    if '__INVERSE_CATEGORY_LABELS__' not in globals():
        labels = rdf_util.create_single_val_dict_from_rdf([utils.get_data_file('files.dbpedia.category_skos')], rdf_util.PREDICATE_PREFLABEL)
        __INVERSE_CATEGORY_LABELS__ = {v: k for k, v in labels.items()}
    return __INVERSE_CATEGORY_LABELS__[label] if label in __INVERSE_CATEGORY_LABELS__ else cat_util.name2category(label)


def get_resources(category: str) -> set:
    """Return all resources of the given category."""
    global __CATEGORY_RESOURCES__
    if '__CATEGORY_RESOURCES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.category_articles')], rdf_util.PREDICATE_SUBJECT, reverse_key=True)
        __CATEGORY_RESOURCES__ = utils.load_or_create_cache('dbpedia_category_resources', initializer)

    return __CATEGORY_RESOURCES__[category]


def get_resource_categories(dbp_resource: str) -> set:
    """Return all categories the given resource is contained in."""
    global __RESOURCE_CATEGORIES__
    if '__RESOURCE_CATEGORIES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.category_articles')], rdf_util.PREDICATE_SUBJECT)
        __RESOURCE_CATEGORIES__ = utils.load_or_create_cache('dbpedia_resource_categories', initializer)

    return __RESOURCE_CATEGORIES__[dbp_resource]


def get_topics(category: str) -> set:
    """Return the topics for the given category."""
    global __TOPICS__
    if '__TOPICS__' not in globals():
        __TOPICS__ = rdf_util.create_multi_val_dict_from_rdf([utils.get_data_file('files.dbpedia.topical_concepts')], rdf_util.PREDICATE_SUBJECT)

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


def get_statistics(category: str) -> dict:
    """Return information about the amounts/frequencies of types and properties of a category's resources."""
    global __CATEGORY_STATISTICS__
    if '__CATEGORY_STATISTICS__' not in globals():
        __CATEGORY_STATISTICS__ = utils.load_or_create_cache('dbpedia_category_statistics', _compute_category_statistics)
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
