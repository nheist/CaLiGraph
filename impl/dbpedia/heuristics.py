"""Implementing heuristics from Töpper et al. 2012 - DBpedia Ontology Enrichment for Inconsistency Detection"""

from typing import Optional, Dict, Set
from collections import defaultdict, Counter
import math
import utils
from functools import cache
from impl.dbpedia.ontology import DbpType, DbpPredicate, DbpOntologyStore
from impl.dbpedia.resource import DbpResource, DbpEntity, DbpResourceStore


DOMAIN_THRESHOLD = .96
RANGE_THRESHOLD = .77


def get_domain(pred: DbpPredicate) -> Optional[DbpType]:
    global __DOMAINS__
    if '__DOMAINS__' not in globals():
        __DOMAINS__ = defaultdict(lambda: None, utils.load_or_create_cache('dbpedia_heuristic_domains', _compute_domains))
    return DbpOntologyStore.instance().get_domain(pred) or __DOMAINS__[pred]


def _compute_domains() -> Dict[DbpPredicate, Optional[DbpType]]:
    return _compute_predicate_types(DbpResourceStore.instance().get_entity_properties(), DOMAIN_THRESHOLD)


def get_range(pred: DbpPredicate) -> Optional[DbpType]:
    global __RANGES__
    if '__RANGES__' not in globals():
        __RANGES__ = defaultdict(lambda: None, utils.load_or_create_cache('dbpedia_heuristic_ranges', _compute_ranges))
    return DbpOntologyStore.instance().get_range(pred) or __RANGES__[pred]


def _compute_ranges() -> Dict[DbpPredicate, Optional[DbpType]]:
    return _compute_predicate_types(DbpResourceStore.instance().get_inverse_entity_properties(), RANGE_THRESHOLD)


def _compute_predicate_types(ent_prop_mapping: Dict[DbpEntity, Dict[DbpPredicate, Set[DbpResource]]], threshold: float) -> Dict[DbpPredicate, Optional[DbpType]]:
    dbo = DbpOntologyStore.instance()

    predicate_type_distribution = defaultdict(Counter)
    for ent in ent_prop_mapping:
        for pred, values in ent_prop_mapping[ent].items():
            triple_count = len(values)
            predicate_type_distribution[pred]['_sum'] += triple_count
            for t in ent.get_transitive_types():
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
                    type_candidates = {t for t in type_candidates if not type_candidates.intersection(dbo.get_transitive_subtypes(t))}

                if len(type_candidates) == 1 or dbo.are_equivalent_types(type_candidates):
                    matching_types[pred] = type_candidates.pop()
    return matching_types


def get_all_disjoint_types(dbp_type: DbpType) -> Set[DbpType]:
    """Return direct and transitive (i.e. disjoint types of parents) disjoint types of `dbp_type`."""
    transitive_dbp_types = DbpOntologyStore.instance().get_transitive_supertypes(dbp_type, include_self=True)
    return {dt for tt in transitive_dbp_types for dt in get_direct_disjoint_types(tt)}


def get_direct_disjoint_types(dbp_type: DbpType) -> Set[DbpType]:
    """Return direct disjoint types of `dbp_type`."""
    global __DISJOINT_TYPES__
    if '__DISJOINT_TYPES__' not in globals():
        type_threshold = utils.get_config('dbpedia.disjointness_threshold')
        __DISJOINT_TYPES__ = utils.load_or_create_cache('dbpedia_heuristic_disjoint_types', lambda: _compute_disjoint_types(type_threshold))
    return __DISJOINT_TYPES__[dbp_type]


def _compute_disjoint_types(type_threshold: float) -> Dict[DbpType, Set[DbpType]]:
    dbo = DbpOntologyStore.instance()
    disjoint_types = defaultdict(set)

    # compute direct disjointnesses
    type_property_weights = _compute_type_property_weights()
    dbp_types = dbo.get_types(include_root=False)
    while len(dbp_types) > 0:
        dbp_type = dbp_types.pop()
        for other_dbp_type in dbp_types:
            if _compute_type_similarity(dbp_type, other_dbp_type, type_property_weights) <= type_threshold:
                disjoint_types[dbp_type].add(other_dbp_type)
                disjoint_types[other_dbp_type].add(dbp_type)

    # remove any disjointnesses that would violate the ontology hierarchy
    # i.e. if two types share a common subtype, they can't be disjoint
    for t in dbo.get_types(include_root=False):
        transitive_types = dbo.get_transitive_supertypes(t, include_self=True)
        for tt in transitive_types:
            disjoint_types[tt] = disjoint_types[tt].difference(transitive_types)

    # add transitive disjointnesses
    disjoint_types = {t: {tdt for dt in dts for tdt in dbo.get_transitive_subtypes(dt, include_self=True)} for t, dts in disjoint_types.items()}

    # make sure that there are no disjointnesses between place and organisation
    dbo_place = dbo.get_class_by_name('Place')
    place_types = dbo.get_transitive_subtypes(dbo_place, include_self=True) | dbo.get_equivalents(dbo_place)
    dbo_organisation = dbo.get_class_by_name('Organisation')
    orga_types = dbo.get_transitive_subtypes(dbo_organisation, include_self=True) | {dbo.get_supertypes(dbo_organisation)}
    for pt in place_types:
        disjoint_types[pt] = disjoint_types[pt].difference(orga_types)
    for ot in orga_types:
        disjoint_types[ot] = disjoint_types[ot].difference(place_types)

    return disjoint_types


def _compute_type_property_weights() -> Dict[DbpType, Dict[DbpPredicate, float]]:
    type_property_weights = defaultdict(lambda: defaultdict(float))

    property_frequencies = _compute_property_frequencies()
    inverse_type_frequencies = _compute_inverse_type_frequencies()
    for dbp_type in DbpOntologyStore.instance().get_types():
        for dbp_pred in inverse_type_frequencies:
            type_property_weights[dbp_type][dbp_pred] = property_frequencies[dbp_type][dbp_pred] * inverse_type_frequencies[dbp_pred]
    return type_property_weights


def _compute_property_frequencies() -> Dict[DbpType, Dict[DbpPredicate, float]]:
    dbr = DbpResourceStore.instance()
    property_frequencies = defaultdict(Counter)
    for res in dbr.get_resources():
        types = res.get_transitive_types()
        for pred, values in res.get_properties().items():
            for t in types:
                property_frequencies[t][pred] += len(values)
    return defaultdict(lambda: defaultdict(float), {t: defaultdict(float, {pred: (1 + math.log(count) if count > 0 else 0) for pred, count in property_frequencies[t].items()}) for t in property_frequencies})


def _compute_inverse_type_frequencies() -> Dict[DbpPredicate, float]:
    dbo = DbpOntologyStore.instance()
    dbr = DbpResourceStore.instance()
    predicate_types = defaultdict(set)
    for res in dbr.get_resources():
        types = res.get_transitive_types()
        for pred in res.get_properties():
            predicate_types[pred].update(types)

    overall_type_count = len(dbo.get_types())
    return {pred: math.log(overall_type_count / (len(predicate_types[pred]) + 1)) for pred in dbo.get_predicates()}


def _compute_type_similarity(type_a: DbpType, type_b: DbpType, type_property_weights: Dict[DbpType, Dict[DbpPredicate, float]]) -> float:
    dbo = DbpOntologyStore.instance()
    if type_a == type_b or type_a in dbo.get_transitive_subtypes(type_b) or type_b in dbo.get_transitive_subtypes(type_a):
        return 1

    numerator = sum(type_property_weights[type_a][pred] * type_property_weights[type_b][pred] for pred in type_property_weights[type_a])
    denominator_a = math.sqrt(sum(type_property_weights[type_a][pred] ** 2 for pred in type_property_weights[type_a]))
    denominator_b = math.sqrt(sum(type_property_weights[type_b][pred] ** 2 for pred in type_property_weights[type_b]))
    return numerator / (denominator_a * denominator_b) if denominator_a * denominator_b > 0 else 1


def is_functional_predicate(pred: DbpPredicate) -> bool:
    """Return True, if the predicate is functional (i.e. a resource has at most one value for the given predicate)."""
    return pred in _find_functional_predicates()


@cache
def _find_functional_predicates() -> set:
    dbo = DbpOntologyStore.instance()
    dbr = DbpResourceStore.instance()

    predicate_resources_count = Counter()
    predicate_nonfunctional_count = Counter()
    for res, properties in dbr.get_entity_properties().items():
        for pred, objects in properties.items():
            predicate_resources_count[pred] += 1
            if len(objects) > 1:
                predicate_nonfunctional_count[pred] += 1

    # if a predicate behaves functional in at least 95% of cases, we assume it to actually be functional
    nonfunctional_preds = {p for p in predicate_resources_count if predicate_nonfunctional_count[p] / predicate_resources_count[p] >= .05}

    return dbo.get_predicates().difference(nonfunctional_preds)
