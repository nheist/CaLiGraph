from collections import defaultdict
import util
import impl.category.store as cat_store
import impl.category.materialize as cat_mat
import impl.dbpedia.store as dbp_store


def get_materialized_statistics(category: str) -> dict:
    global __MATERIALIZED_CATEGORY_STATISTICS__
    if '__MATERIALIZED_CATEGORY_STATISTICS__' not in globals():
        __MATERIALIZED_CATEGORY_STATISTICS__ = util.load_or_create_cache('dbpedia_materialized_category_statistics', lambda: _compute_category_statistics(use_materialized_graph=True))
    return __MATERIALIZED_CATEGORY_STATISTICS__[category]


def get_statistics(category: str) -> dict:
    global __CATEGORY_STATISTICS__
    if '__CATEGORY_STATISTICS__' not in globals():
        __CATEGORY_STATISTICS__ = util.load_or_create_cache('dbpedia_category_statistics', _compute_category_statistics)
    return __CATEGORY_STATISTICS__[category]


def _compute_category_statistics(use_materialized_graph: bool = False) -> dict:
    util.get_logger().info('Computing category statistics..')
    category_statistics = {}
    for cat in cat_store.get_all_cats():
        type_counts = defaultdict(int)
        property_counts = defaultdict(int)
        property_counts_inv = defaultdict(int)
        predicate_counts = defaultdict(int)
        predicate_counts_inv = defaultdict(int)

        resources = cat_mat.get_materialized_resources(cat) if use_materialized_graph else cat_store.get_resources(cat)
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
