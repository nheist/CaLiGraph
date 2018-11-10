import util
from typing import Optional
import caligraph.dbpedia.store as dbp_store
from collections import defaultdict
import math

# implementing heuristics from Töpper et al. 2012 - DBpedia Ontology Enrichment for Inconsistency Detection
DOMAIN_THRESHOLD = .96
DISJOINT_THRESHOLD = .17


def get_domain(dbp_predicate) -> Optional[str]:
    global __DOMAINS__
    if '__DOMAINS__' not in globals():
        __DOMAINS__ = util.load_or_create_cache('dbpedia_heuristic_domains', _compute_domains)
    return __DOMAINS__[dbp_predicate] if dbp_predicate in __DOMAINS__ else None


def _compute_domains() -> dict:
    predicate_type_distribution = defaultdict(lambda: defaultdict(int))

    resource_property_mapping = dbp_store.get_resource_property_mapping()
    for r in resource_property_mapping:
        for pred in resource_property_mapping[r]:
            triple_count = len(resource_property_mapping[r][pred])
            predicate_type_distribution[pred]['_sum'] += triple_count
            for t in dbp_store.get_transitive_types(r):
                predicate_type_distribution[pred][t] += triple_count

    predicate_domains = {}
    for pred in predicate_type_distribution:
        t_sum = predicate_type_distribution[pred]['_sum']
        t_scores = {t: t_count / t_sum for t, t_count in predicate_type_distribution[pred].items() if t != '_sum'}
        if t_scores:
            t_score_max = max(t_scores.values())
            if t_score_max >= DOMAIN_THRESHOLD:
                valid_domains = {t for t, t_score in t_scores.items() if t_score == t_score_max}
                if len(valid_domains) > 1:
                    valid_domains = {t for t in valid_domains if not valid_domains.intersection(dbp_store.get_transitive_subtypes(t))}

                if len(valid_domains) == 1 or dbp_store.are_equivalent_types(valid_domains):
                    predicate_domains[pred] = valid_domains.pop()

    return predicate_domains


def get_disjoint_types(dbp_type) -> set:
    global __DISJOINT_TYPES__
    if '__DISJOINT_TYPES__' not in globals():
        __DISJOINT_TYPES__ = util.load_or_create_cache('dbpedia_heuristic_disjoint_types', _compute_disjoint_types)
    return __DISJOINT_TYPES__[dbp_type]


def _compute_disjoint_types() -> dict:
    disjoint_types = defaultdict(set)

    type_property_weights = _compute_type_property_weights()
    dbp_types = {t for types in dbp_store._get_resource_type_mapping().values() for t in types}
    while len(dbp_types) > 0:
        dbp_type = dbp_types.pop()
        for other_dbp_type in dbp_types:
            if _compute_type_similarity(dbp_type, other_dbp_type, type_property_weights) <= DISJOINT_THRESHOLD:
                disjoint_types[dbp_type].add(other_dbp_type)
                disjoint_types[other_dbp_type].add(dbp_type)

    return disjoint_types  # todo: check whether transitive types are included


def _compute_type_property_weights() -> dict:
    type_property_weights = {}

    inverse_type_frequencies = {pred: _compute_inverse_type_frequency(pred) for pred in dbp_store.get_all_predicates()}
    for dbp_type in dbp_store.get_all_types():
        for dbp_pred in inverse_type_frequencies:
            type_property_weights[dbp_type][dbp_pred] = _compute_property_frequency(dbp_type, dbp_pred) * inverse_type_frequencies[dbp_pred]
    return type_property_weights


def _compute_property_frequency(dbp_type: str, dbp_predicate: str) -> int:
    pf = len({r for r in dbp_store.get_resources() if dbp_type in dbp_store.get_transitive_types(r) and dbp_predicate in dbp_store.get_properties(r)})
    return 1 + math.log2(pf) if pf > 0 else 0


def _compute_inverse_type_frequency(dbp_predicate: str) -> float:
    types = dbp_store.get_all_types()
    predicate_types = len({t for t in types if any(dbp_predicate in dbp_store.get_properties(r) for r in dbp_store.get_resources())})
    return math.log2(len(types) / predicate_types)


def _compute_type_similarity(type_a: str, type_b: str, type_property_weights: dict) -> float:
    numerator = sum(type_property_weights[type_a][pred] * type_property_weights[type_b][pred] for pred in type_property_weights[type_a])
    denominator_a = math.sqrt(sum([w ** 2 for pred in type_property_weights[type_a] for w in type_property_weights[type_a][pred]]))
    denominator_b = math.sqrt(sum([w ** 2 for pred in type_property_weights[type_b] for w in type_property_weights[type_a][pred]]))
    return numerator / (denominator_a * denominator_b)

