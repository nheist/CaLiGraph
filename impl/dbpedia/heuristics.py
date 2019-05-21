import util
from typing import Optional
import impl.util.rdf as rdf_util
import impl.dbpedia.store as dbp_store
from collections import defaultdict
import math

# implementing heuristics from Töpper et al. 2012 - DBpedia Ontology Enrichment for Inconsistency Detection
DOMAIN_THRESHOLD = .96
RANGE_THRESHOLD = .77
DISJOINT_THRESHOLD = .17
MAX_DISJOINT_THRESHOLD = .5


def get_domain(dbp_predicate: str) -> Optional[str]:
    global __DOMAINS__
    if '__DOMAINS__' not in globals():
        __DOMAINS__ = defaultdict(lambda: None, util.load_or_create_cache('dbpedia_heuristic_domains', _compute_domains))
    return dbp_store.get_domain(dbp_predicate) or __DOMAINS__[dbp_predicate]


def _compute_domains() -> dict:
    return _compute_predicate_types(dbp_store.get_resource_property_mapping(), DOMAIN_THRESHOLD)


def get_range(dbp_predicate: str) -> Optional[str]:
    global __RANGES__
    if '__RANGES__' not in globals():
        __RANGES__ = defaultdict(lambda: None, util.load_or_create_cache('dbpedia_heuristic_ranges', _compute_ranges))
    return dbp_store.get_range(dbp_predicate) or __RANGES__[dbp_predicate]


def _compute_ranges() -> dict:
    return _compute_predicate_types(dbp_store.get_inverse_resource_property_mapping(), RANGE_THRESHOLD)


def _compute_predicate_types(resource_property_mapping: dict, threshold: float) -> dict:
    predicate_type_distribution = defaultdict(lambda: defaultdict(int))
    for r in resource_property_mapping:
        for pred, values in resource_property_mapping[r].items():
            triple_count = len(values)
            predicate_type_distribution[pred]['_sum'] += triple_count
            for t in dbp_store.get_transitive_types(r):
                predicate_type_distribution[pred][t] += triple_count

    matching_types = {}
    for pred in predicate_type_distribution:
        t_sum = predicate_type_distribution[pred]['_sum']
        t_scores = {t: t_count / t_sum for t, t_count in predicate_type_distribution[pred].items() if t != '_sum'}
        if t_scores:
            t_score_max = max(t_scores.values())
            if t_score_max >= threshold:
                type_candidates = {t for t, t_score in t_scores.items() if t_score == t_score_max}
                if len(type_candidates) > 1:
                    type_candidates = {t for t in type_candidates if not type_candidates.intersection(dbp_store.get_transitive_subtypes(t))}

                if len(type_candidates) == 1 or dbp_store.are_equivalent_types(type_candidates):
                    matching_types[pred] = type_candidates.pop()
    return matching_types


def get_disjoint_types(dbp_type, threshold=DISJOINT_THRESHOLD) -> set:
    global __DISJOINT_TYPES__
    if '__DISJOINT_TYPES__' not in globals():
        __DISJOINT_TYPES__ = util.load_or_create_cache('dbpedia_heuristic_disjoint_types', _compute_disjoint_types)
    return {tp for tp, sim in __DISJOINT_TYPES__[dbp_type] if sim <= threshold}


def _compute_disjoint_types() -> dict:
    disjoint_types = defaultdict(set)

    type_property_weights = _compute_type_property_weights()
    dbp_types = dbp_store.get_all_types().difference({rdf_util.CLASS_OWL_THING})
    while len(dbp_types) > 0:
        dbp_type = dbp_types.pop()
        for other_dbp_type in dbp_types:
            threshold = _compute_type_similarity(dbp_type, other_dbp_type, type_property_weights)
            if threshold <= MAX_DISJOINT_THRESHOLD:
                disjoint_types[dbp_type].add((other_dbp_type, threshold))
                disjoint_types[other_dbp_type].add((dbp_type, threshold))
    return disjoint_types


def _compute_type_property_weights() -> dict:
    type_property_weights = defaultdict(lambda: defaultdict(float))

    property_frequencies = _compute_property_frequencies()
    inverse_type_frequencies = _compute_inverse_type_frequencies()
    for dbp_type in dbp_store.get_all_types():
        for dbp_pred in inverse_type_frequencies:
            type_property_weights[dbp_type][dbp_pred] = property_frequencies[dbp_type][dbp_pred] * inverse_type_frequencies[dbp_pred]
    return type_property_weights


def _compute_property_frequencies() -> dict:
    property_frequencies = defaultdict(lambda: defaultdict(int))
    for r in dbp_store.get_resources():
        types = dbp_store.get_transitive_types(r)
        for pred, values in dbp_store.get_properties(r).items():
            for t in types:
                property_frequencies[t][pred] += len(values)
    return defaultdict(lambda: defaultdict(float), {t: defaultdict(float, {pred: (1 + math.log2(count) if count > 0 else 0) for pred, count in property_frequencies[t].items()}) for t in property_frequencies})


def _compute_inverse_type_frequencies() -> dict:
    predicate_types = defaultdict(set)
    for r in dbp_store.get_resources():
        for pred in dbp_store.get_properties(r):
            predicate_types[pred].update(dbp_store.get_transitive_types(r))

    overall_type_count = len(dbp_store.get_all_types())
    return {pred: math.log2(overall_type_count / (len(predicate_types[pred]) + 1)) for pred in dbp_store.get_all_predicates()}


def _compute_type_similarity(type_a: str, type_b: str, type_property_weights: dict) -> float:
    numerator = sum(type_property_weights[type_a][pred] * type_property_weights[type_b][pred] for pred in type_property_weights[type_a])
    denominator_a = math.sqrt(sum([type_property_weights[type_a][pred] ** 2 for pred in type_property_weights[type_a]]))
    denominator_b = math.sqrt(sum([type_property_weights[type_b][pred] ** 2 for pred in type_property_weights[type_b]]))
    return numerator / (denominator_a * denominator_b) if denominator_a * denominator_b > 0 else 0

