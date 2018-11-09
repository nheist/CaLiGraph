from collections import defaultdict
import datetime
import caligraph.category.store as cat_store
from caligraph.category.graph import CategoryGraph
import caligraph.dbpedia.store as dbp_store
import util
import pandas as pd
import random
from typing import Tuple
from collections import namedtuple

Property = namedtuple('Property', 'p o')
Fact = namedtuple('Fact', 's p o')

PROPERTY_INGOING = 'ingoing'
PROPERTY_OUTGOING = 'outgoing'

MIN_CAT_PROPERTY_COUNT = 1
MIN_CAT_PROPERTY_FREQ = .2

# todo: --- GENERAL ---
# todo: use purity of types instead of disjointness for domain/range constraints ( -> HEIKO Paper)
# todo: evaluation a) hold-out set (DONE); b) instance-based manual/mturk (DONE); c) category-based manual/mturk
# todo: Profiling: Welche Subject-Types/Properties findet man gut/schlecht (Aufstellung bis FREITAG, was wichtig ist)

# todo: --- REFACTOR ---
# todo: 'Property' einbauen


def _get_property_count(resources: set, property_mapping: dict) -> dict:
    cat_property_count = defaultdict(int)
    for res in resources:
        for prop, values in property_mapping[res].items():
            for val in values:
                cat_property_count[(prop, val)] += 1
    return cat_property_count


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
                other_facts = {Fact(s=r, p=pred, o=val) for val in new_values.difference(existing_values)}
                false_facts.update(other_facts) if dbp_store.is_functional(pred) else unknown_facts.update(other_facts)
            else:
                unknown_facts.update({Fact(s=r, p=pred, o=val) for val in new_values})

    return true_facts, false_facts, unknown_facts


def _compute_metrics(true_facts: set, false_facts: set) -> Tuple[float, float]:
    existing_facts_count = sum({len(vals) for r in dbp_store.get_resources() for vals in dbp_store.get_properties(r).values()})
    true_facts_count, false_facts_count = len(true_facts), len(false_facts)

    precision = true_facts_count / (true_facts_count + false_facts_count)
    recall = true_facts_count / existing_facts_count
    return precision, recall


def _create_evaluation_dump(unknown_facts: set, size: int, relation_type: str):
    filename = 'results/relations-v6-{}-{}_{}_{}.csv'.format(relation_type, size, MIN_CAT_PROPERTY_COUNT, int(MIN_CAT_PROPERTY_FREQ*100))

    size = len(unknown_facts) if len(unknown_facts) < size else size
    df = pd.DataFrame(data=random.sample(unknown_facts, size), columns=['sub', 'pred', 'obj'])
    df.to_csv(filename, index=False, encoding='utf-8')


def evaluate_parameters():
    evaluation_results = []
    for min_count in [1, 2, 3, 4, 5]:
        for min_freq in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
            util.get_logger().info('Evaluating params: {} / {:.3f}'.format(min_count, min_freq))
            result = evaluate_category_relations(min_count, min_freq)
            result['count_min'] = min_count
            result['freq_min'] = min_freq
            evaluation_results.append(result)
    results = pd.DataFrame(data=evaluation_results)
    results.to_csv('results/relations-v6_parameter-optimization.csv', index=False, encoding='utf-8')


def evaluate_category_relations(min_cat_property_count: int = MIN_CAT_PROPERTY_COUNT, min_cat_property_freq: float = MIN_CAT_PROPERTY_FREQ) -> dict:
    categories = CategoryGraph.create_from_dbpedia().remove_unconnected().nodes
    result = {}

    util.get_logger().info('-- OUTGOING PROPERTIES --')
    invalid_pred_types = defaultdict(set, {p: dbp_store.get_disjoint_types(dbp_store.get_domain(p)) for p in dbp_store.get_all_predicates()})
    out_property_assignments = _assign_resource_properties(categories, dbp_store.get_resource_property_mapping(), invalid_pred_types, min_cat_property_count, min_cat_property_freq)
    out_true, out_false, out_unknown = _split_assignments(out_property_assignments)
    out_precision, out_recall = _compute_metrics(out_true, out_false)

    util.get_logger().info('Precision: {:.3f}; Recall: {:.3f}; New-Count: {}'.format(out_precision, out_recall, len(out_unknown)))
    _create_evaluation_dump(out_unknown, 200, PROPERTY_OUTGOING)
    result[f'{PROPERTY_OUTGOING}_precision'] = out_precision
    result[f'{PROPERTY_OUTGOING}_recall'] = out_recall
    result[f'{PROPERTY_OUTGOING}_F1'] = (2 * out_precision * out_recall) / (out_precision + out_recall)
    result[f'{PROPERTY_OUTGOING}_new-count'] = len(out_unknown)

    util.get_logger().info('-- INGOING PROPERTIES --')
    invalid_pred_types = defaultdict(set, {p: dbp_store.get_disjoint_types(dbp_store.get_range(p)) for p in dbp_store.get_all_predicates()})
    inverse_ingoing_property_assignments = _assign_resource_properties(categories, dbp_store.get_inverse_resource_property_mapping(), invalid_pred_types, min_cat_property_count, min_cat_property_freq)
    in_property_assignments = defaultdict(lambda: defaultdict(set))
    for sub in inverse_ingoing_property_assignments:
        for pred in inverse_ingoing_property_assignments[sub]:
            for obj in inverse_ingoing_property_assignments[sub][pred]:
                in_property_assignments[obj][pred].add(sub)

    in_true, in_false, in_unknown = _split_assignments(in_property_assignments)
    in_precision, in_recall = _compute_metrics(in_true, in_false)

    util.get_logger().info('Precision: {:.3f}; Recall: {:.3f}; New-Count: {}'.format(in_precision, in_recall, len(in_unknown)))
    _create_evaluation_dump(in_unknown, 200, PROPERTY_INGOING)
    result[f'{PROPERTY_INGOING}_precision'] = in_precision
    result[f'{PROPERTY_INGOING}_recall'] = in_recall
    result[f'{PROPERTY_INGOING}_F1'] = (2 * in_precision * in_recall) / (in_precision + in_recall)
    result[f'{PROPERTY_INGOING}_new-count'] = len(in_unknown)

    return result


def _assign_resource_properties(categories: set, property_mapping: dict, invalid_pred_types: dict, min_cat_property_count: int, min_cat_property_freq: float) -> dict:
    starttime = datetime.datetime.now().replace(microsecond=0)
    cat_counter = 0
    property_counter = 0
    instance_counter = 0

    property_assignments = defaultdict(lambda: defaultdict(set))
    for idx, cat in enumerate(categories):
        resources = cat_store.get_resources(cat)
        cat_property_count = _get_property_count(resources, property_mapping)
        cat_property_freq = {p: p_count / len(resources) for p, p_count in cat_property_count.items()}

        valid_properties = {p for p in cat_property_count
                            if cat_property_count[p] >= min_cat_property_count
                            and cat_property_freq[p] >= min_cat_property_freq
                            and any(surf in cat_store.get_label(cat).lower() for surf in dbp_store.get_surface_forms(p[1]))}

        if valid_properties:
            cat_counter += 1
            util.get_logger().debug('=' * 20)
            util.get_logger().debug('Category: {}'.format(cat[37:]))

            for prop in valid_properties:
                predicate, val = prop

                # filter out reflexive relations
                valid_resources = {r for r in resources if r != val}
                # filter out list pages
                valid_resources = {r for r in valid_resources if 'List_of_' not in r}
                # filter out functional relations with existing values
                valid_resources = {r for r in valid_resources if not dbp_store.get_properties(r)[predicate]} if dbp_store.is_functional(predicate) else valid_resources
                # filter out invalid domains / ranges
                invalid_types = invalid_pred_types[predicate]
                valid_resources = {r for r in valid_resources if not invalid_types.intersection(dbp_store.get_types(r))} if invalid_types else valid_resources

                if valid_resources:
                    for r in valid_resources:
                        property_assignments[r][predicate].add(val)

                    property_counter += 1
                    instance_counter += len(resources) - cat_property_count[prop]
                    util.get_logger().debug('Property: {} ({} / {} / {:.3f})'.format(prop, len(resources), cat_property_count[prop], cat_property_freq[prop]))

                if len(resources) > len(valid_resources):
                    util.get_logger().debug('Removed {} invalid fact assignments due to domain/range violation.'.format( len(resources) - len(valid_resources)))

        if idx % 1000 == 0:
            util.get_logger().debug('=' * 20)
            util.get_logger().debug('Processed {} of {} categories in {}.'.format(idx, len(categories), (datetime.datetime.now().replace(microsecond=0) - starttime)))

    util.get_logger().info(f'CATS: {cat_counter} -- PROPERTIES: {property_counter} -- INSTANCES: {instance_counter}')
    return property_assignments
