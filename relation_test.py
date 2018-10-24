from collections import defaultdict
import datetime
import caligraph.category.store as cat_store
from caligraph.category.graph import CategoryGraph
import caligraph.dbpedia.store as dbp_store
import util


MIN_CAT_PROPERTY_COUNT = 3  # OK
MIN_CAT_PROPERTY_FREQ = .6  # OK
# MIN_CAT_PROPERTY_DIFF = .8 # -> exclude for now; if we have other property values, we dismiss.
MAX_OVERALL_PROPERTY_FREQ = 1  # might not even need that


# todo: comparison of relation-instances with sibling-categories and parent-categories (if they have it, too, it can't be that special to this category)
# todo: use surface-forms of objects to find similar object values within categories
# todo: also look for incoming relation instances!
# todo: treat functional (single-valued) vs. non-functional (multi-valued) relations differently
# todo: evaluation not via wikidata -> a) hold-out set; b) instance-based manual/mturk c) category-based manual/mturk
# todo: check whether a relation can be "generalized" over the complete category by checking whether other categories with this relation have instances with differing values
# --> not if we find categories where we have equally distributed values, but others (e.g. categories where we have sth. like 80/20

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


def evaluate_category_relations():
    starttime = datetime.datetime.now().replace(microsecond=0)
    cat_counter = 0
    property_counter = 0
    instance_counter = 0

    categories = CategoryGraph.create_from_dbpedia().remove_unconnected().nodes
    for idx, cat in enumerate(categories):
        resources = cat_store.get_resources(cat)
        cat_property_count = _get_property_count(resources)
        cat_property_value_count = _get_property_value_count(cat_property_count.keys())
        cat_property_freq = {p: p_count / len(resources) for p, p_count in cat_property_count.items()}
        overall_property_freq = {p: (dbp_store.get_property_frequency_distribution(p[0])[p[1]] - p_count + 1) / (dbp_store.get_property_frequency_distribution(p[0])['_sum'] - p_count + 1) for p, p_count in cat_property_count.items()}

        valid_properties = {p for p in cat_property_count if cat_property_count[p] >= MIN_CAT_PROPERTY_COUNT and cat_property_freq[p] >= MIN_CAT_PROPERTY_FREQ and overall_property_freq[p] <= MAX_OVERALL_PROPERTY_FREQ}

        if valid_properties:
            cat_counter += 1
            util.get_logger().debug('='*20)
            util.get_logger().debug('Category: {}'.format(cat[37:]))

            for p in valid_properties:
                property_counter += 1
                instance_counter += len(resources) - cat_property_count[p]
                util.get_logger().debug('Property: {} ({} / {} / {:.3f} / {:.3f})'.format(p, len(resources), cat_property_count[p], cat_property_freq[p], overall_property_freq[p]))

        if idx % 1000 == 0:
            util.get_logger().debug('=' * 20)
            util.get_logger().debug('Processed {} of {} categories in {}.'.format(idx, len(categories), (datetime.datetime.now().replace(microsecond=0) - starttime)))

    util.get_logger().debug(f'CATS: {cat_counter} -- PROPERTIES: {property_counter} -- INSTANCES: {instance_counter}')
