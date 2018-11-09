import util
from typing import Optional
import caligraph.dbpedia.store as dbp_store
from collections import defaultdict

# implementing heuristics from Töpper et al. 2012 - DBpedia Ontology Enrichment for Inconsistency Detection
DOMAIN_THRESHOLD = .90
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
            for t in dbp_store.get_transitive_types(r):
                predicate_type_distribution[pred][t] += 1

    predicate_domains = {}
    for pred in predicate_type_distribution:
        t_sum = sum(predicate_type_distribution[pred].values())
        t_scores = {t: t_count / t_sum for t, t_count in predicate_type_distribution[pred].items()}
        t_score_max = max(t_scores.values())
        print(f'{pred}: {t_score_max}')
        if t_score_max >= DOMAIN_THRESHOLD:
            valid_domains = {t for t, t_score in t_scores.items() if t_score == t_score_max}
            if len(valid_domains) > 1:
                valid_domains = {t for t in valid_domains if not valid_domains.intersection(dbp_store.get_transitive_subtypes(t))}

            if len(valid_domains) > 1:
                print(f'SOMETHING IS GOING WRONG WHILE RETRIEVING DOMAIN FOR {pred}: {valid_domains}')
            predicate_domains[pred] = valid_domains.pop()

    return predicate_domains


def get_disjoint_types(dbp_type) -> set:
    global __DISJOINT_TYPES__
    if '__DISJOINT_TYPES__' not in globals():
        __DISJOINT_TYPES__ = util.load_or_create_cache('dbpedia_heuristic_disjoint_types', _compute_disjoint_types)
    return __DISJOINT_TYPES__[dbp_type]


def _compute_disjoint_types() -> dict:
    pass
    # dbp_types = dbp_store.get_types()
