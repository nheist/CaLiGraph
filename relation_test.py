from collections import defaultdict
import datetime
import caligraph.category.store as cat_store
from caligraph.category.graph import CategoryGraph
import caligraph.dbpedia.store as dbp_store
import util
import pandas as pd
import random
from nltk.stem import PorterStemmer


MIN_CAT_PROPERTY_COUNT = 1
MIN_CAT_PROPERTY_FREQ = .2
# classifier erstmal zweitrangig
# todo: train classifier to predict these values?
# todo: attribute - do i find a surface form in the category name
# todo: attribute - negative example for disjoint domain

# todo: --- GENERAL ---
# todo: use purity of types instead of disjointness for domain/range constraints ( -> HEIKO Paper)
# todo: nach F1 evaluieren und optimale Parameterkonstellation finden
# todo: evaluation a) hold-out set (DONE); b) instance-based manual/mturk (DONE); c) category-based manual/mturk
# todo: Profiling: Welche Subject-Types/Properties findet man gut/schlecht (Aufstellung bis FREITAG, was wichtig ist)


def _get_property_count(resources: set, property_mapping: dict) -> dict:
    cat_property_count = defaultdict(int)
    for res in resources:
        for prop, values in property_mapping[res].items():
            for val in values:
                cat_property_count[(prop, val)] += 1
    return cat_property_count


def _get_property_value_count(property_tuples) -> dict:
    property_value_count = defaultdict(set)
    for pred, val in property_tuples:
        property_value_count[pred].add(val)
    return {pred: len(vals) for pred, vals in property_value_count.items()}


def _compute_metrics(resource_property_assignments: dict):
    existing_assignments = 0
    new_assignments = 0
    correct_assignments = 0
    incorrect_assignments = 0

    for r in dbp_store.get_resources():
        existing_properties = dbp_store.get_properties(r)
        existing_assignments += sum({len(vals) for vals in existing_properties.values()})

        for pred, new_values in resource_property_assignments[r].items():
            existing_values = existing_properties[pred]
            new_assignments += len(new_values)
            correct_assignments += len(new_values.intersection(existing_values))
            incorrect_assignments += len(new_values.difference(existing_values)) if existing_values else 0

    precision = correct_assignments / (correct_assignments + incorrect_assignments)
    recall = correct_assignments / existing_assignments
    count_new = new_assignments - correct_assignments - incorrect_assignments
    return precision, recall, count_new


def _create_evaluation_dump(resource_property_assignments: dict, size: int, relation_type: str):
    filename = 'results/relations-v5-{}_{}_{}_{}.csv'.format(size, relation_type, MIN_CAT_PROPERTY_COUNT, int(MIN_CAT_PROPERTY_FREQ*100))
    unclear_assignments = [(r, pred, val) for r in resource_property_assignments for pred in resource_property_assignments[r] for val in resource_property_assignments[r][pred] if not dbp_store.get_properties(r)[pred]]

    size = len(unclear_assignments) if len(unclear_assignments) < size else size
    df = pd.DataFrame(data=random.sample(unclear_assignments, size), columns=['sub', 'pred', 'val'])
    df.to_csv(filename, index=False, encoding='utf-8')


def evaluate_parameters():
    evaluation_results = []
    for min_cat_property_count in [1, 2, 3, 4, 5]:
        for min_cat_property_freq in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
            util.get_logger().info('Evaluating params: {} / {:.3f}'.format(min_cat_property_count, min_cat_property_freq))
            precision, recall, count_new = evaluate_category_relations(min_cat_property_count, min_cat_property_freq)
            util.get_logger().info('Precision: {:.3f}; Recall: {:.3f}'.format(precision, recall))
            evaluation_results.append({'min_cat_property_count': min_cat_property_count, 'min_cat_property_freq': min_cat_property_freq, 'precision': precision, 'recall': recall, 'count_new': count_new})
    results = pd.DataFrame(data=evaluation_results)
    results.to_csv('results/relations-v4_parameter-optimization.csv')


def evaluate_category_relations(min_cat_property_count: int = MIN_CAT_PROPERTY_COUNT, min_cat_property_freq: float = MIN_CAT_PROPERTY_FREQ) -> tuple:
    categories = CategoryGraph.create_from_dbpedia().remove_unconnected().nodes

    # util.get_logger().info('-- OUTGOING PROPERTIES --')
    invalid_pred_types = defaultdict(set, {p: dbp_store.get_disjoint_types(dbp_store.get_domain(p)) for p in dbp_store.get_all_predicates()})
    outgoing_property_assignments = _assign_resource_properties(categories, dbp_store.get_resource_property_mapping(), invalid_pred_types, min_cat_property_count, min_cat_property_freq)

    precision_out, recall_out, count_out = _compute_metrics(outgoing_property_assignments)
    util.get_logger().debug('Precision: {:.3f}; Recall: {:.3f}; New-Count: {}'.format(precision_out, recall_out, count_out))
    _create_evaluation_dump(outgoing_property_assignments, 200, 'out')

    # util.get_logger().info('-- INGOING PROPERTIES --')
    # invalid_pred_types = defaultdict(set, {p: dbp_store.get_disjoint_types(dbp_store.get_range(p)) for p in dbp_store.get_all_predicates()})
    # inverse_ingoing_property_assignments = _assign_resource_properties(categories, dbp_store.get_inverse_resource_property_mapping(), invalid_pred_types)
    # ingoing_property_assignments = defaultdict(lambda: defaultdict(set))
    # for sub in inverse_ingoing_property_assignments:
    #     for pred in inverse_ingoing_property_assignments[sub]:
    #         for obj in inverse_ingoing_property_assignments[sub][pred]:
    #             ingoing_property_assignments[obj][pred].add(sub)
    #
    # precision_in, recall_in = _compute_metrics(ingoing_property_assignments)
    # util.get_logger().debug('Precision: {:.3f}; Recall: {:.3f}'.format(precision_in, recall_in))
    # _create_evaluation_dump(ingoing_property_assignments, 200, 'in')

    return precision_out, recall_out, count_out


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

        ps = PorterStemmer()
        stemmed_cat_label = ' '.join({ps.stem(word) for word in cat_store.get_label(cat).split(' ')})
        valid_properties = {p for p in cat_property_count
                            if cat_property_count[p] >= min_cat_property_count
                            and cat_property_freq[p] >= min_cat_property_freq
                            and any(ps.stem(surf) in stemmed_cat_label for surf in dbp_store.get_surface_forms(p[1]))}

        if valid_properties:
            cat_counter += 1
            util.get_logger().debug('=' * 20)
            util.get_logger().debug('Category: {}'.format(cat[37:]))

            for prop in valid_properties:
                predicate, val = prop
                invalid_types = invalid_pred_types[predicate]

                # filter out reflexive relations
                valid_resources = {r for r in resources if r != val}
                # filter out list pages
                valid_resources = {r for r in valid_resources if 'List_of_' not in r}
                # filter out functional relations with existing values
                valid_resources = {r for r in valid_resources if not dbp_store.get_properties(r)[predicate]} if dbp_store.is_functional(predicate) else valid_resources
                # filter out invalid domains / ranges
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
