import util
import functools
import pandas as pd
from typing import Tuple
from collections import defaultdict
from impl.category.graph import CategoryGraph
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.category.base as cat_base
import impl.category.store as cat_store
from sklearn.ensemble import RandomForestClassifier
from rdflib import Graph, URIRef, Literal


def extract_axioms_and_relation_assertions():
    category_axioms = get_category_axioms()
    category_axioms.to_csv(util.get_results_file('results.cataxioms.category_axioms'), index=False)

    relation_assertions = _compute_new_relation_assertions(category_axioms)
    _serialize_relation_assertions(relation_assertions[relation_assertions['lcwa-true']], util.get_results_file('results.cataxioms.relation_assertions_lcwa_true'))
    _serialize_relation_assertions(relation_assertions[~relation_assertions['lcwa-true']], util.get_results_file('results.cataxioms.relation_assertions_lcwa_false'))


def _compute_new_relation_assertions(category_axioms: pd.DataFrame) -> pd.DataFrame:
    relation_assertions = set()
    for _, row in category_axioms.iterrows():
        cat, pred, val, is_inv = row['cat'], row['pred'], row['val'], row['is_inv']
        subjects = cat_base.get_taxonomic_category_graph().get_materialized_resources(cat) if util.get_config('category.axioms.use_materialized_category_graph') else cat_store.get_resources(cat)
        for sub in subjects:
            if is_inv:
                sub, val = val, sub
            properties = dbp_store.get_properties(sub)
            if pred in properties and val in properties[pred]:
                continue  # assertion already in KG
            if sub == val:
                continue  # no reflexive assertions
            if 'List_of_' in sub:
                continue  # no assertions for organisational resources
            relation_assertions.add((sub, pred, val, pred not in properties))

    return pd.DataFrame(data=list(relation_assertions), columns=['sub', 'pred', 'val', 'lcwa-true'])


def _serialize_relation_assertions(data: pd.DataFrame, destination: str):
    rdf_graph = Graph()
    for _, row in data.iterrows():
        sub, pred, obj = row['sub'], row['pred'], row['val']
        if dbp_util.is_dbp_resource(obj) or dbp_util.is_dbp_type(obj):
            rdf_graph.add((URIRef(sub), URIRef(pred), URIRef(obj)))
        else:
            rdf_graph.add((URIRef(sub), URIRef(pred), Literal(obj)))
    rdf_graph.serialize(destination=destination, format='nt')


def get_category_axioms() -> pd.DataFrame:
    candidate_axioms = util.load_or_create_cache('cataxioms_candidates', _compute_candidate_axioms)
    X, y = _create_goldstandard(candidate_axioms)
    axiom_estimator = RandomForestClassifier(n_estimators=350, max_depth=5, class_weight={0: util.get_config('category.axioms.pos_weight'), 1: 1}).fit(X, y)

    return _compute_correct_axioms(axiom_estimator, candidate_axioms)


def _compute_correct_axioms(axiom_estimator: RandomForestClassifier, candidate_axioms: pd.DataFrame) -> pd.DataFrame:
    candidate_axioms['prediction'] = axiom_estimator.predict(candidate_axioms)
    return candidate_axioms[candidate_axioms['prediction'] == True].reset_index()[['cat', 'pred', 'val', 'is_inv']]


def _create_goldstandard(category_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    labels = pd.read_csv(util.get_data_file('files.evaluation.category_properties'), index_col=['cat', 'pred', 'val', 'is_inv'])
    goldstandard = pd.merge(category_data, labels, how='inner', on=['cat', 'pred', 'val', 'is_inv'])
    y = goldstandard['label']
    X = goldstandard.drop(columns='label')
    return X, y


def _compute_candidate_axioms() -> pd.DataFrame:
    categories = CategoryGraph.create_from_dbpedia().remove_unconnected().nodes
    categories = {cat for cat in categories if 'List_of_' not in cat and 'Lists_of_' not in cat}
    type_freqs = util.load_or_create_cache('cataxioms_type_frequencies', functools.partial(_compute_type_frequencies, categories))

    property_mapping = dbp_store.get_resource_property_mapping()
    property_counts, property_freqs, predicate_instances = util.load_or_create_cache('cataxioms_property_stats', functools.partial(_compute_property_stats, categories, property_mapping))

    inv_property_mapping = dbp_store.get_inverse_resource_property_mapping()
    inv_property_counts, inv_property_freqs, inv_predicate_instances = util.load_or_create_cache('cataxioms_inverse_property_stats', functools.partial(_compute_property_stats, categories, inv_property_mapping))

    outgoing_data = _get_candidates(categories, property_counts, property_freqs, predicate_instances, type_freqs, _get_invalid_domains(), False)
    ingoing_data = _get_candidates(categories, inv_property_counts, inv_property_freqs, inv_predicate_instances, type_freqs, _get_invalid_ranges(), True)
    return pd.DataFrame(data=[*outgoing_data, *ingoing_data]).set_index(['cat', 'pred', 'val', 'is_inv'])


def _compute_type_frequencies(categories: set) -> dict:
    type_counts = defaultdict(functools.partial(defaultdict, int))
    type_frequencies = defaultdict(functools.partial(defaultdict, float))

    for idx, cat in enumerate(categories):
        if idx % 1000:
            util.get_logger().debug(f'Computing type frequencies for categories: {idx} / {len(categories)}')
        resources = cat_base.get_taxonomic_category_graph().get_materialized_resources(cat) if util.get_config('category.axioms.use_materialized_category_graph') else cat_store.get_resources(cat)
        for res in resources:
            for t in dbp_store.get_transitive_types(res):
                type_counts[cat][t] += 1
        type_frequencies[cat] = defaultdict(float, {t: t_count / len(resources) for t, t_count in type_counts[cat].items()})

    return type_frequencies


def _compute_property_stats(categories: set, property_mapping: dict) -> Tuple[dict, dict, dict]:
    property_counts = defaultdict(functools.partial(defaultdict, int))
    property_frequencies = defaultdict(functools.partial(defaultdict, float))
    predicate_instances = defaultdict(functools.partial(defaultdict, int))

    for idx, cat in enumerate(categories):
        if idx % 1000:
            util.get_logger().debug(f'Computing property stats for categories: {idx} / {len(categories)}')
        resources = cat_base.get_taxonomic_category_graph().get_materialized_resources(cat) if util.get_config('category.axioms.use_materialized_category_graph') else cat_store.get_resources(cat)
        for res in resources:
            resource_values = set()
            for pred, values in property_mapping[res].items():
                predicate_instances[cat][pred] += 1
                for val in {dbp_store.resolve_redirect(v) for v in values}:
                    property_counts[cat][(pred, val)] += 1
                    resource_values.add(val)
        property_frequencies[cat] = defaultdict(float, {prop: p_count / len(resources) for prop, p_count in property_counts[cat].items()})

    return property_counts, property_frequencies, predicate_instances


def _get_invalid_domains() -> dict:
    return defaultdict(set, {p: dbp_store.get_disjoint_types(dbp_store.get_domain(p)) for p in dbp_store.get_all_predicates()})


def _get_invalid_ranges() -> dict:
    return defaultdict(set, {p: dbp_store.get_disjoint_types(dbp_store.get_range(p)) for p in dbp_store.get_all_predicates()})


def _get_candidates(categories: set, property_counts: dict, property_freqs: dict, predicate_instances: dict, type_freqs: dict, invalid_pred_types: dict, is_inv: bool) -> list:
    conceptual_cats = cat_base.get_conceptual_category_graph().nodes
    candidates = []
    for cat in categories:
        for prop in property_counts[cat].keys():
            pred, val = prop
            lex_score = dbp_store.get_surface_score(val, cat_store.get_label(cat))
            if lex_score > 0:
                candidates.append({
                    'cat': cat,
                    'pred': pred,
                    'val': val,
                    'is_inv': int(is_inv),
                    'pv_count': property_counts[cat][prop],
                    'pv_freq': property_freqs[cat][prop],
                    'lex_score': lex_score,
                    'conflict_score': sum([type_freqs[cat][t] for t in invalid_pred_types[pred]]) + (1 - (property_counts[cat][prop] / predicate_instances[cat][pred])),
                    'conceptual_category': int(cat in conceptual_cats)
                })
    return candidates
