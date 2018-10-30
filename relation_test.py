from collections import defaultdict
import datetime
import caligraph.category.store as cat_store
from caligraph.category.graph import CategoryGraph
import caligraph.dbpedia.store as dbp_store
import util
import pandas as pd
import random


MIN_CAT_PROPERTY_COUNT = 3
MIN_CAT_PROPERTY_FREQ = .6


# todo: comparison of relation-instances with sibling-categories and parent-categories (if they have it, too, it can't be that special to this category)
# todo: !!! use surface-forms of objects to find similar object values within categories --> not relevant enough
# todo: !!! also look for incoming relation instances!
# todo: !!! treat functional (single-valued) vs. non-functional (multi-valued) relations differently
# todo: evaluation not via wikidata -> a) hold-out set; b) instance-based manual/mturk c) category-based manual/mturk
# todo: check whether a relation can be "generalized" over the complete category by checking whether other categories with this relation have instances with differing values
# --> not if we find categories where we have equally distributed values, but others (e.g. categories where we have sth. like 80/20
# todo: exclude lists
# todo: use purity of types instead of disjointness for domain/range constraints

def _get_property_count(resources: set) -> dict:
    cat_property_count = defaultdict(int)
    for res in resources:
        for prop, values in dbp_store.get_properties(res).items():
            for val in values:
                cat_property_count[(prop, val)] += 1
    return cat_property_count


def _get_property_value_count(property_tuples) -> dict:
    property_value_count = defaultdict(set)
    for pred, val in property_tuples:
        property_value_count[pred].add(val)
    return {pred: len(vals) for pred, vals in property_value_count.items()}


def _compute_metrics(resource_property_assignments: dict):
    all_assignments = 0
    correct_assignments = 0
    incorrect_assignments = 0
    for r in dbp_store.get_resources():
        for pred, actual_values in dbp_store.get_properties(r).items():
            assigned_values = resource_property_assignments[r][pred]
            all_assignments += len(actual_values)
            correct_assignments += len(assigned_values.intersection(actual_values))
            incorrect_assignments += len(assigned_values.difference(actual_values)) if pred in resource_property_assignments[r] else 0

    precision = correct_assignments / (correct_assignments + incorrect_assignments)
    recall = correct_assignments / all_assignments
    return precision, recall


def _create_evaluation_dump(resource_property_assignments: dict, size: int, relation_type: str):
    filename = 'results/relations-{}-v2_{}_{}_{}.csv'.format(size, relation_type, MIN_CAT_PROPERTY_COUNT, int(MIN_CAT_PROPERTY_FREQ*100))
    unclear_assignments = [(r, pred, val) for r in resource_property_assignments for pred in resource_property_assignments[r] for val in resource_property_assignments[r][pred] if pred not in dbp_store.get_properties(r)]

    df = pd.DataFrame(data=random.sample(unclear_assignments, size), columns=['sub', 'pred', 'val'])
    df.to_csv(filename, index=False, encoding='utf-8')


def evaluate_category_relations():
    categories = CategoryGraph.create_from_dbpedia().remove_unconnected().nodes

    util.get_logger().info('-- OUTGOING PROPERTIES --')
    outgoing_resource_property_assignments = _assign_resource_properties(categories, False)

    precision_out, recall_out = _compute_metrics(outgoing_resource_property_assignments)
    util.get_logger().debug('Precision: {:.3f}; Recall: {:.3f}'.format(precision_out, recall_out))
    _create_evaluation_dump(outgoing_resource_property_assignments, 200, 'out')

    util.get_logger().info('-- INCOMING PROPERTIES --')
    outgoing_resource_property_assignments = _assign_resource_properties(categories, True)  # todo: implement inverse


def _assign_resource_properties(categories: set, invert_properties: bool):
    starttime = datetime.datetime.now().replace(microsecond=0)
    cat_counter = 0
    property_counter = 0
    instance_counter = 0

    resource_property_assignments = defaultdict(lambda: defaultdict(set))
    for idx, cat in enumerate(categories):
        resources = cat_store.get_resources(cat)
        cat_property_count = _get_property_count(resources)
        cat_property_freq = {p: p_count / len(resources) for p, p_count in cat_property_count.items()}

        valid_properties = {p for p in cat_property_count if
                            cat_property_count[p] >= MIN_CAT_PROPERTY_COUNT and cat_property_freq[
                                p] >= MIN_CAT_PROPERTY_FREQ}

        if valid_properties:
            cat_counter += 1
            util.get_logger().debug('=' * 20)
            util.get_logger().debug('Category: {}'.format(cat[37:]))

            for prop in valid_properties:
                predicate, val = prop
                domain = dbp_store.get_domain(predicate)
                invalid_domain_types = dbp_store.get_disjoint_types(domain) if domain else set()
                range = dbp_store.get_range(predicate)  # todo: for ingoing properties
                invalid_range_types = dbp_store.get_disjoint_types(range) if range else set()
                valid_resources = {r for r in resources if not invalid_domain_types.intersection(dbp_store.get_types(r))} if invalid_domain_types else resources
                if valid_resources:
                    for r in valid_resources:
                        resource_property_assignments[r][predicate].add(val)

                    property_counter += 1
                    instance_counter += len(resources) - cat_property_count[prop]
                    util.get_logger().debug('Property: {} ({} / {} / {:.3f})'.format(prop, len(resources), cat_property_count[prop], cat_property_freq[prop]))

                if len(resources) > len(valid_resources):
                    util.get_logger().debug('Removed {} invalid fact assignments due to domain/range violation.'.format( len(resources) - len(valid_resources)))

        if idx % 1000 == 0:
            util.get_logger().debug('=' * 20)
            util.get_logger().debug('Processed {} of {} categories in {}.'.format(idx, len(categories), (datetime.datetime.now().replace(microsecond=0) - starttime)))

    util.get_logger().debug(f'CATS: {cat_counter} -- PROPERTIES: {property_counter} -- INSTANCES: {instance_counter}')
    return resource_property_assignments
