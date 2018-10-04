from collections import Counter, defaultdict
import datetime
import caligraph.category.store as cat_store
import caligraph.dbpedia.store as dbp_store
import util


MIN_CAT_PROPERTY_COUNT = 3  # OK
MIN_CAT_PROPERTY_FREQ = .8  # OK
# MIN_CAT_PROPERTY_DIFF = .8 # -> exclude for now; if we have other property values, we dismiss.
MAX_OVERALL_PROPERTY_FREQ = 1  # might not even need that


def _get_property_value_count(property_tuples) -> dict:
    property_value_count = defaultdict(set)
    for prop, val in property_tuples:
        property_value_count[prop].add(val)
    return {prop: len(vals) for prop, vals in property_value_count.items()}


def evaluate_category_relations():
    starttime = datetime.datetime.now().replace(microsecond=0)
    cat_counter = 0
    property_counter = 0
    instance_counter = 0

    categories = cat_store.get_all_cats()
    for idx, cat in enumerate(categories):
        resources = cat_store.get_resources(cat)
        cat_property_count = sum([Counter(dbp_store.get_properties(res)) for res in resources], Counter())
        cat_property_value_count = _get_property_value_count(cat_property_count.keys())
        cat_property_freq = {p: p_count / len(resources) for p, p_count in cat_property_count.items()}
        overall_property_freq = {p: (dbp_store.get_property_frequency_distribution(p[0])[p[1]] - p_count + 1) / (
                    dbp_store.get_property_frequency_distribution(p[0])['_sum'] - p_count + 1) for p, p_count in
                                 cat_property_count.items()}

        valid_properties = {p for p in cat_property_count if
                            cat_property_value_count[p[0]] > 1 and cat_property_count[p] >= MIN_CAT_PROPERTY_COUNT and
                            cat_property_freq[p] >= MIN_CAT_PROPERTY_FREQ and overall_property_freq[
                                p] <= MAX_OVERALL_PROPERTY_FREQ}

        if valid_properties:
            cat_counter += 1

        for p in valid_properties:
            property_counter += 1
            instance_counter += len(resources) - cat_property_count[p]
        #    print('='*20)
        #    print('Category: {}'.format(cat[37:]))
        #    print('Property: {}'.format(p))
        #    print('ResourceCount: {} -- CatPropertyCount: {}'.format(len(resources), cat_property_count[p]))
        #    print('CatPropertyFreq: {:.3f} -- OverallPropertyFreq: {:.3f}'.format(cat_property_freq[p], overall_property_freq[p]))

        if idx % 1000 == 0:
            util.get_logger().debug('=' * 20)
            util.get_logger().debug('Processed {} of {} categories in {}.'.format(idx, len(categories), (datetime.datetime.now().replace(microsecond=0) - starttime)))
            util.get_logger().debug('FOUND RELATIONS FOR -> Categories: {} -- Properties: {} -- Instances: {}'.format(cat_counter, property_counter, instance_counter))

    util.get_logger().debug(f'CATS: {cat_counter} -- PROPERTIES: {property_counter} -- INSTANCES: {instance_counter}')
