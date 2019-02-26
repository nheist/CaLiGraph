from . import util as cat_util
import impl.util.rdf as rdf_util
import util


def get_all_cats() -> set:
    global __CATEGORIES__
    if '__CATEGORIES__' not in globals():
        initializer = lambda: set(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_TYPE))
        __CATEGORIES__ = util.load_or_create_cache('dbpedia_categories', initializer)

    return __CATEGORIES__


def get_usable_cats() -> set:
    return {cat for cat in get_all_cats() if is_usable(cat)}


def is_usable(category: str) -> bool:
    # filtering maintenance categories
    if category in get_maintenance_cats():
        return False
    # filtering other administrative categories
    indicators = ['wikipedia', 'wikiprojects', 'lists', 'redirects', 'mediawiki', 'template', 'user', 'portal', 'categories', 'articles', 'pages', 'navigational']
    lower_category = category.lower()
    if any(indicator in lower_category for indicator in indicators):
        return False
    return True


def get_label(category: str) -> str:
    global __LABELS__
    if '__LABELS__' not in globals():
        __LABELS__ = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_SKOS_LABEL)

    return __LABELS__[category] if category in __LABELS__ else cat_util.category2name(category)


def get_resources(category: str) -> set:
    global __RESOURCES__
    if '__RESOURCES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.article_categories')], rdf_util.PREDICATE_SUBJECT, reverse_key=True)
        __RESOURCES__ = util.load_or_create_cache('dbpedia_category_resources', initializer)

    return __RESOURCES__[category]


def get_resource_to_cats_mapping() -> dict:
    if '__RESOURCE_CATEGORIES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.article_categories')], rdf_util.PREDICATE_SUBJECT)
        global __RESOURCE_CATEGORIES__
        __RESOURCE_CATEGORIES__ = util.load_or_create_cache('dbpedia_resource_categories', initializer)

    return __RESOURCE_CATEGORIES__


def get_children(category: str) -> set:
    global __CHILDREN__
    if '__CHILDREN__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_BROADER, reverse_key=True)
        __CHILDREN__ = util.load_or_create_cache('dbpedia_category_children', initializer)

    return __CHILDREN__[category].difference({category})


def get_redirects(category: str) -> set:
    global __REDIRECTS__
    if '__REDIRECTS__' not in globals():
        __REDIRECTS__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.category_redirects')], rdf_util.PREDICATE_REDIRECTS)

    return __REDIRECTS__[category]


def get_topics(category: str) -> set:
    global __TOPICS__
    if '__TOPICS__' not in globals():
        __TOPICS__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.topical_concepts')], rdf_util.PREDICATE_SUBJECT)

    return __TOPICS__[category]


def get_maintenance_cats() -> set:
    global __MAINTENANCE_CATS__
    if '__MAINTENANCE_CATS__' not in globals():
        __MAINTENANCE_CATS__ = set(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.maintenance_categories')], rdf_util.PREDICATE_TYPE))

    return __MAINTENANCE_CATS__
