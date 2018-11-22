from collections import defaultdict
import caligraph.category.store as cat_store
import caligraph.category.base as cat_base
from caligraph.category.graph import CategoryGraph
import caligraph.dbpedia.store as dbp_store
import caligraph.dbpedia.util as dbp_util
import caligraph.dbpedia.heuristics as dbp_heuristics
import util
import pandas as pd
import numpy as np
import random
from typing import Tuple
from collections import namedtuple
import functools
import operator
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

COMPUTE_BASELINE = True
USE_HEURISTIC_CONSTRAINTS = True  # HC
USE_RESOLVED_REDIRECTS = True  # RR

CategoryProperty = namedtuple('CategoryProperty', 'cat pred obj prob count inv')


class BaselineEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.estimator = GaussianNB()

    def fit(self, X, y, sample_weight=None):
        self.estimator.fit(X[['freq']], y, sample_weight)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[['freq']])

    def predict(self, X):
        return self.estimator.predict(X[['freq']])


def evaluate_classification_category_relations():
    data = get_category_data()
    X, y = get_goldstandard(data)

    estimators = {'Baseline': BaselineEstimator(), 'Naive Bayes': GaussianNB(), 'k-NN': KNeighborsClassifier(), 'SVM': SVC(random_state=42), 'Random Forest': RandomForestClassifier(random_state=42), 'XG-Boost': XGBClassifier(random_state=42), 'Neural Net': MLPClassifier(random_state=42)}
    scoring = {'F1': 'f1', 'P': 'precision', 'R': 'recall', 'ACC': 'accuracy', 'ROC': 'roc_auc'}
    for e_name, e in estimators.items():
        scores = cross_validate(e, X, y, scoring=scoring, cv=StratifiedKFold(n_splits=10, random_state=42), n_jobs=10)
        f1, prec, rec, acc, roc = scores['test_F1'], scores['test_P'], scores['test_R'], scores['test_ACC'], scores['test_ROC']
        util.get_logger().info('{}: F1={:.2f} P={:.2f} R={:.2f} ACC={:.2f} ROC={:.2f}'.format(e_name, *[np.mean(val) * 100 for val in [f1, prec, rec, acc, roc]]))


def get_category_data(version=None) -> pd.DataFrame:
    category_data = util.load_or_create_cache('relations_category_data', _compute_category_data, version=version)

    if 'domain' in category_data.columns:
        category_data = category_data.join(pd.get_dummies(category_data['domain'], prefix='domain')).drop(columns='domain')
    if 'range' in category_data.columns:
        category_data = category_data.join(pd.get_dummies(category_data['range'], prefix='range')).drop(columns='range')

    return category_data


def get_goldstandard(category_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = _load_labels()
    X = pd.merge(y.to_frame(), category_data, how='left', on=['cat', 'pred', 'obj', 'is_inv']).drop(columns='label')
    return X, y


def _compute_category_data() -> pd.DataFrame:
    categories = CategoryGraph.create_from_dbpedia().remove_unconnected().nodes
    type_freqs = util.load_or_create_cache('relations_type_frequencies', functools.partial(_compute_type_frequencies, categories))

    property_mapping = dbp_store.get_resource_property_mapping()
    property_counts, property_freqs, predicate_instances = util.load_or_create_cache('relations_property_stats', functools.partial(_compute_property_stats, categories, property_mapping), version=('RR' if USE_RESOLVED_REDIRECTS else None))
    surface_property_values = util.load_or_create_cache('relations_surface_property_values', functools.partial(_compute_surface_property_values, categories, property_mapping), version=('RR' if USE_RESOLVED_REDIRECTS else None))

    inv_property_mapping = dbp_store.get_inverse_resource_property_mapping()
    inv_property_counts, inv_property_freqs, inv_predicate_instances = util.load_or_create_cache('relations_inverse_property_stats', functools.partial(_compute_property_stats, categories, inv_property_mapping), version=('RR' if USE_RESOLVED_REDIRECTS else None))
    inv_surface_property_values = util.load_or_create_cache('relations_inverse_surface_property_values', functools.partial(_compute_surface_property_values, categories, inv_property_mapping), version=('RR' if USE_RESOLVED_REDIRECTS else None))

    outgoing_data = _get_samples(categories, property_counts, property_freqs, predicate_instances, type_freqs, _get_invalid_domains(), surface_property_values, False)
    ingoing_data = _get_samples(categories, inv_property_counts, inv_property_freqs, inv_predicate_instances, type_freqs, _get_invalid_ranges(), inv_surface_property_values, True)
    return pd.DataFrame(data=[*outgoing_data, *ingoing_data]).set_index(['cat', 'pred', 'obj', 'is_inv'], drop=False).drop(columns=['cat', 'pred', 'obj'])


def _get_samples(categories: set, property_counts: dict, property_freqs: dict, predicate_instances: dict, type_freqs: dict, invalid_pred_types: dict, surface_property_values: dict, is_inv: bool) -> list:
    conceptual_cats = cat_base.get_conceptual_category_graph().nodes
    samples = []
    for cat in categories:
        for prop in property_counts[cat].keys():
            pred, val = prop
            if surface_property_values[cat][val]:
                samples.append({
                    'cat': cat,
                    'pred': pred,
                    'obj': val,
                    'is_inv': int(is_inv),
                    'count': property_counts[cat][prop],
                    'freq': property_freqs[cat][prop],
                    'surf': surface_property_values[cat][val],
                    'neg': sum([type_freqs[cat][t] for t in invalid_pred_types[pred]]) + (1 - (property_counts[cat][(pred, val)] / predicate_instances[cat][pred])),
                    'is_functional': int(dbp_store.is_functional(pred)),
                    'is_conceptual': int(cat in conceptual_cats),
                    'is_object': int(dbp_util.is_dbp_resource(val)),
                    'domain': dbp_heuristics.get_domain(pred),
                    'range': dbp_heuristics.get_range(pred)
                })
    return samples


def _load_labels() -> pd.Series:
    return pd.read_csv(util.get_data_file('files.evaluation.category_properties'), index_col=['cat', 'pred', 'obj', 'is_inv'])['label']


def evaluate_probabilistic_category_relations():
    categories = CategoryGraph.create_from_dbpedia().remove_unconnected().nodes
    type_freqs = util.load_or_create_cache('relations_type_frequencies', functools.partial(_compute_type_frequencies, categories))

    property_mapping = dbp_store.get_resource_property_mapping()
    property_counts, property_freqs, predicate_instances = util.load_or_create_cache('relations_property_stats', functools.partial(_compute_property_stats, categories, property_mapping), version=('RR' if USE_RESOLVED_REDIRECTS else None))
    surface_property_values = util.load_or_create_cache('relations_surface_property_values', functools.partial(_compute_surface_property_values, categories, property_mapping), version=('RR' if USE_RESOLVED_REDIRECTS else None))

    inv_property_mapping = dbp_store.get_inverse_resource_property_mapping()
    inv_property_counts, inv_property_freqs, inv_predicate_instances = util.load_or_create_cache('relations_inverse_property_stats', functools.partial(_compute_property_stats, categories, inv_property_mapping), version=('RR' if USE_RESOLVED_REDIRECTS else None))
    inv_surface_property_values = util.load_or_create_cache('relations_inverse_surface_property_values', functools.partial(_compute_surface_property_values, categories, inv_property_mapping), version=('RR' if USE_RESOLVED_REDIRECTS else None))

    in_probabilities = _compute_property_probabilites(categories, inv_property_counts, inv_property_freqs, inv_predicate_instances, type_freqs, _get_invalid_ranges(), inv_surface_property_values, True)
    out_probabilities = _compute_property_probabilites(categories, property_counts, property_freqs, predicate_instances, type_freqs, _get_invalid_domains(), surface_property_values, False)
    relations_probabilities = out_probabilities | in_probabilities
    util.update_cache('relations_probabilities', relations_probabilities)
    return relations_probabilities


def _compute_property_probabilites(categories: set, property_counts: dict, property_freqs: dict, predicate_instances: dict, type_freqs: dict, invalid_pred_types: dict, surface_property_values: dict, is_inv: bool) -> set:
    cat_properties = set()
    for idx, cat in enumerate(categories):
        util.get_logger().debug(f'checking category {cat} ({idx}/{len(categories)})..')
        for pred, val in property_freqs[cat].keys():
            util.get_logger().debug(f'checking predicate {pred} and value {val}..')
            util.get_logger().debug(f'surface forms for {val}: {surface_property_values[cat][val]}')
            p = property_freqs[cat][(pred, val)]
            if COMPUTE_BASELINE:
                if surface_property_values[cat][val]:
                    util.get_logger().debug(f'{cat} / {pred} / {val} is correct!')
                    cat_properties.add(CategoryProperty(cat=cat, pred=pred, obj=val, prob=p, count=property_counts[cat][(pred, val)], inv=is_inv))
            else:
                c_given_p = surface_property_values[cat][val]
                if p * c_given_p > 0:
                    c_given_not_p = sum([type_freqs[cat][t] for t in invalid_pred_types[pred]]) + (1 - (property_counts[cat][(pred, val)] / predicate_instances[cat][pred]))
                    c = c_given_p * p + c_given_not_p * (1 - p)
                    p_given_c = c_given_p * p / c if c > 0 else 0
                    if p_given_c > 0:
                        cat_properties.add(CategoryProperty(cat=cat, pred=pred, obj=val, prob=p_given_c, count=property_counts[cat][(pred, val)], inv=is_inv))

    return cat_properties


def _get_invalid_domains():
    predicates = dbp_store.get_all_predicates()
    if USE_HEURISTIC_CONSTRAINTS:
        return defaultdict(set, {p: dbp_heuristics.get_disjoint_types(dbp_heuristics.get_domain(p)) for p in predicates})
    else:
        return defaultdict(set, {p: dbp_store.get_disjoint_types(dbp_store.get_domain(p)) for p in predicates})


def _get_invalid_ranges():
    predicates = dbp_store.get_all_predicates()
    if USE_HEURISTIC_CONSTRAINTS:
        return defaultdict(set, {p: dbp_heuristics.get_disjoint_types(dbp_heuristics.get_range(p)) for p in predicates})
    else:
        return defaultdict(set, {p: dbp_store.get_disjoint_types(dbp_store.get_range(p)) for p in predicates})


def _compute_property_stats(categories: set, property_mapping: dict) -> Tuple[dict, dict, dict]:
    property_counts = defaultdict(functools.partial(defaultdict, int))
    property_frequencies = defaultdict(functools.partial(defaultdict, float))
    predicate_instances = defaultdict(functools.partial(defaultdict, int))

    for cat in categories:
        resources = cat_store.get_resources(cat)
        for res in resources:
            for pred, values in property_mapping[res].items():
                predicate_instances[cat][pred] += 1
                values = {dbp_store.resolve_redirect(v) for v in values} if USE_RESOLVED_REDIRECTS else values
                for val in values:
                    property_counts[cat][(pred, val)] += 1
        property_frequencies[cat] = defaultdict(float, {prop: p_count / len(resources) for prop, p_count in property_counts[cat].items()})

    return property_counts, property_frequencies, predicate_instances


def _compute_type_frequencies(categories: set) -> dict:
    type_counts = defaultdict(functools.partial(defaultdict, int))
    type_frequencies = defaultdict(functools.partial(defaultdict, float))

    for cat in categories:
        resources = cat_store.get_resources(cat)
        for res in resources:
            for t in dbp_store.get_transitive_types(res):
                type_counts[cat][t] += 1
        type_frequencies[cat] = defaultdict(float, {t: t_count / len(resources) for t, t_count in type_counts[cat].items()})

    return type_frequencies


def _compute_surface_property_values(categories: set, property_mapping: dict) -> dict:
    surface_property_values = defaultdict(functools.partial(defaultdict, float))
    for cat in categories:
        possible_values = {val for r in cat_store.get_resources(cat) for values in property_mapping[r].values() for val in values}
        for val in possible_values:
            cat_label = cat_store.get_label(cat).lower()
            target_val = dbp_store.resolve_redirect(val) if USE_RESOLVED_REDIRECTS else val
            surface_forms = {**dbp_store.get_surface_forms(val), **dbp_store.get_surface_forms(target_val)}
            total_mentions = sum(surface_forms.values())
            for surf, mentions in sorted(surface_forms.items(), key=operator.itemgetter(1), reverse=True):
                if surf in cat_label:
                    surface_property_values[cat][target_val] = mentions / total_mentions
                    break
    return surface_property_values


# --- DEPRECATED ---
Fact = namedtuple('Fact', 's p o')

PROPERTY_INGOING = 'ingoing'
PROPERTY_OUTGOING = 'outgoing'

MIN_PROPERTY_COUNT = 1
MIN_PROPERTY_FREQ = .1
MAX_INVALID_TYPE_COUNT = 1
MAX_INVALID_TYPE_FREQ = .1


def evaluate_parameters():
    evaluation_results = []
    for min_count in [1]:  # [1, 2, 3, 4, 5]:
        for min_freq in [.1]:  # [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
            for max_invalid_type_count in [1, 2, 3, 4, 5]:
                for max_invalid_type_freq in [0, .1, .2, .3, .4, .5]:
                    util.get_logger().info('Evaluating params: {} / {:.3f} / {} / {:.3f}'.format(min_count, min_freq, max_invalid_type_count, max_invalid_type_freq))
                    result = evaluate_category_relations(min_count, min_freq, max_invalid_type_count, max_invalid_type_freq)
                    result['0_property_count_min'] = min_count
                    result['1_property_freq_min'] = min_freq
                    result['2_invalid_type_count_max'] = max_invalid_type_count
                    result['3_invalid_type_freq_max'] = max_invalid_type_freq
                    evaluation_results.append(result)
    results = pd.DataFrame(data=evaluation_results)
    results.to_csv('results/relations-v9_parameter-optimization.csv', index=False, encoding='utf-8')


def evaluate_category_relations(min_count: int = MIN_PROPERTY_COUNT, min_freq: float = MIN_PROPERTY_FREQ, max_invalid_type_count: int = MAX_INVALID_TYPE_COUNT, max_invalid_type_freq: float = MAX_INVALID_TYPE_FREQ) -> dict:
    categories = CategoryGraph.create_from_dbpedia().remove_unconnected().nodes
    property_counts, property_freqs, predicate_instances = util.load_or_create_cache('relations_property_stats', functools.partial(_compute_property_stats, categories, dbp_store.get_resource_property_mapping()))
    inverse_property_counts, inverse_property_freqs, inverse_predicate_instances = util.load_or_create_cache('relations_inverse_property_stats', functools.partial(_compute_property_stats, categories, dbp_store.get_inverse_resource_property_mapping()))
    type_counts, type_freqs = util.load_or_create_cache('relations_type_stats', functools.partial(_compute_type_frequencies, categories))
    invalid_predicate_types = _get_invalid_domains()
    surface_property_values = util.load_or_create_cache('relations_surface_property_values', functools.partial(_compute_surface_property_values, categories))
    result = {}

    util.get_logger().info('-- OUTGOING PROPERTIES --')
    out_cat_assignments, out_fact_assignments = _compute_assignments(categories, property_counts, type_counts, invalid_predicate_types['domain'], surface_property_values, min_count, min_freq, max_invalid_type_count, max_invalid_type_freq)
    out_true, out_false, out_unknown = _split_assignments(out_fact_assignments)
    out_precision, out_recall = _compute_metrics(out_true, out_false)

    util.get_logger().info('Precision: {:.3f}; Recall: {:.3f}; New-Count: {}'.format(out_precision, out_recall, len(out_unknown)))
    _create_evaluation_dump({(cat, pred, obj) for cat in out_cat_assignments for pred in out_cat_assignments[cat] for obj in out_cat_assignments[cat][pred]}, 200, f'cats-{PROPERTY_OUTGOING}', min_count, min_freq, max_invalid_type_count, max_invalid_type_freq)
    _create_evaluation_dump(out_unknown, 200, f'facts-{PROPERTY_OUTGOING}', min_count, min_freq, max_invalid_type_count, max_invalid_type_freq)
    _create_evaluation_dump(out_false, 200, f'false-facts-{PROPERTY_OUTGOING}', min_count, min_freq, max_invalid_type_count, max_invalid_type_freq)
    result.update({
        f'{PROPERTY_OUTGOING}_cat-count': len(out_cat_assignments),
        f'{PROPERTY_OUTGOING}_pred-count': sum([len(out_cat_assignments[cat]) for cat in out_cat_assignments]),
        f'{PROPERTY_OUTGOING}_axiom-count': sum([len(out_cat_assignments[cat][pred]) for cat in out_cat_assignments for pred in out_cat_assignments[cat]]),
        f'{PROPERTY_OUTGOING}_inst-count': sum([len(out_fact_assignments[cat][pred]) for cat in out_fact_assignments for pred in out_fact_assignments[cat]]),
        f'{PROPERTY_OUTGOING}_true-count': len(out_true),
        f'{PROPERTY_OUTGOING}_false-count': len(out_false),
        f'{PROPERTY_OUTGOING}_new-count': len(out_unknown),
        f'{PROPERTY_OUTGOING}_precision': out_precision,
        f'{PROPERTY_OUTGOING}_recall': out_recall,
        f'{PROPERTY_OUTGOING}_F1': (2 * out_precision * out_recall) / (out_precision + out_recall)
    })

    util.get_logger().info('-- INGOING PROPERTIES --')
    in_cat_assignments, in_inverse_fact_assignments = _compute_assignments(categories, inverse_property_counts, type_counts, invalid_predicate_types['range'], surface_property_values, min_count, min_freq, max_invalid_type_count, max_invalid_type_freq)
    in_fact_assignments = defaultdict(lambda: defaultdict(set))
    for sub in in_inverse_fact_assignments:
        for pred in in_inverse_fact_assignments[sub]:
            for obj in in_inverse_fact_assignments[sub][pred]:
                in_fact_assignments[obj][pred].add(sub)

    in_true, in_false, in_unknown = _split_assignments(in_fact_assignments)
    in_precision, in_recall = _compute_metrics(in_true, in_false)

    util.get_logger().info('Precision: {:.3f}; Recall: {:.3f}; New-Count: {}'.format(in_precision, in_recall, len(in_unknown)))
    _create_evaluation_dump({(cat, pred, obj) for cat in in_cat_assignments for pred in in_cat_assignments[cat] for obj in in_cat_assignments[cat][pred]}, 200, f'cats-{PROPERTY_INGOING}', min_count, min_freq, max_invalid_type_count, max_invalid_type_freq)
    _create_evaluation_dump(in_unknown, 200, f'facts-{PROPERTY_INGOING}', min_count, min_freq, max_invalid_type_count, max_invalid_type_freq)
    _create_evaluation_dump(in_false, 200, f'false-facts-{PROPERTY_INGOING}', min_count, min_freq, max_invalid_type_count, max_invalid_type_freq)
    result.update({
        f'{PROPERTY_INGOING}_cat-count': len(in_cat_assignments),
        f'{PROPERTY_INGOING}_pred-count': sum([len(in_cat_assignments[cat]) for cat in in_cat_assignments]),
        f'{PROPERTY_INGOING}_axiom-count': sum([len(in_cat_assignments[cat][pred]) for cat in in_cat_assignments for pred in in_cat_assignments[cat]]),
        f'{PROPERTY_INGOING}_inst-count': sum([len(in_fact_assignments[cat][pred]) for cat in in_fact_assignments for pred in in_fact_assignments[cat]]),
        f'{PROPERTY_INGOING}_true-count': len(in_true),
        f'{PROPERTY_INGOING}_false-count': len(in_false),
        f'{PROPERTY_INGOING}_new-count': len(in_unknown),
        f'{PROPERTY_INGOING}_precision': in_precision,
        f'{PROPERTY_INGOING}_recall': in_recall,
        f'{PROPERTY_INGOING}_F1': (2 * in_precision * in_recall) / (in_precision + in_recall)
    })

    return result


def _compute_assignments(categories: set, property_counts: dict, type_counts: dict, invalid_pred_types: dict, surface_property_values: dict, min_property_count: int, min_property_freq: float, max_invalid_type_count: int, max_invalid_type_freq: float) -> Tuple[dict, dict]:
    util.get_logger().debug('computing assignments..')
    cat_assignments = defaultdict(lambda: defaultdict(set))
    fact_assignments = defaultdict(lambda: defaultdict(set))
    for idx, cat in enumerate(categories):
        util.get_logger().debug(f'checking category {cat} ({idx}/{len(categories)})..')
        resources = cat_store.get_resources(cat)
        property_count = property_counts[cat]
        property_freq = {p: p_count / len(resources) for p, p_count in property_count.items()}
        type_count = type_counts[cat]
        type_freq = defaultdict(float, {t: t_count / len(resources) for t, t_count in type_count.items()})

        util.get_logger().debug('computing valid properties..')
        valid_properties = {p for p in property_count
                            if property_count[p] >= min_property_count
                            and property_freq[p] >= min_property_freq
                            and all(type_count[t] <= max_invalid_type_count for t in invalid_pred_types[p[0]])
                            and all(type_freq[t] <= max_invalid_type_freq for t in invalid_pred_types[p[0]])
                            and p[1] in surface_property_values[cat]}

        if valid_properties:
            util.get_logger().debug('=' * 20)
            util.get_logger().debug('Category: {}'.format(cat[37:]))

            for prop in valid_properties:
                predicate, val = prop

                cat_assignments[cat][predicate].add(val)
                # filter out reflexive relations
                valid_resources = {r for r in resources if r != val}
                # filter out list pages
                valid_resources = {r for r in valid_resources if 'List_of_' not in r}
                # filter out functional relations with existing values
                valid_resources = {r for r in valid_resources if not dbp_store.get_properties(r)[predicate]} if dbp_store.is_functional(predicate) else valid_resources
                # filter out invalid domains / ranges
                invalid_types = {tt for t in invalid_pred_types[predicate] for tt in dbp_store.get_transitive_subtype_closure(t)}
                valid_resources = {r for r in valid_resources if not invalid_types.intersection(dbp_store.get_types(r))} if invalid_types else valid_resources

                if valid_resources:
                    for r in valid_resources:
                        fact_assignments[r][predicate].add(val)
                    util.get_logger().debug('Property: {} ({} / {} / {:.3f})'.format(prop, len(resources), property_count[prop], property_freq[prop]))

    return cat_assignments, fact_assignments


def _split_assignments(property_assignments: dict) -> Tuple[set, set, set]:
    true_facts = set()
    false_facts = set()
    unknown_facts = set()

    for r in dbp_store.get_resources():
        existing_properties = dbp_store.get_properties(r)
        for pred, new_values in property_assignments[r].items():
            existing_values = existing_properties[pred]
            if existing_values:
                true_facts.update({Fact(s=r, p=pred, o=val) for val in new_values.intersection(existing_values)})
                false_facts.update({Fact(s=r, p=pred, o=val) for val in new_values.difference(existing_values)})
            else:
                unknown_facts.update({Fact(s=r, p=pred, o=val) for val in new_values})

    return true_facts, false_facts, unknown_facts


def _compute_metrics(true_facts: set, false_facts: set) -> Tuple[float, float]:
    existing_facts_count = sum([len(vals) for r in dbp_store.get_resources() for vals in dbp_store.get_properties(r).values()])
    true_facts_count, false_facts_count = len(true_facts), len(false_facts)

    precision = true_facts_count / (true_facts_count + false_facts_count)
    recall = true_facts_count / existing_facts_count
    return precision, recall


def _create_evaluation_dump(data: set, size: int, relation_type: str, min_count: int, min_freq: float, max_invalid_type_count: int, max_invalid_type_freq: float):
    filename = 'results/relations-v9-{}-{}_{}_{}_{}_{}.csv'.format(relation_type, size, min_count, int(min_freq*100), max_invalid_type_count, int(max_invalid_type_freq*100))

    size = len(data) if len(data) < size else size
    df = pd.DataFrame(data=random.sample(data, size), columns=['sub', 'pred', 'obj'])
    df.to_csv(filename, index=False, encoding='utf-8')
