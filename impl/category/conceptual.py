import util
import impl.util.nlp as nlp_util
import impl.category.store as cat_store


def is_conceptual_category(category: str) -> bool:
    """Return true, if the category is conceptual (i.e. has a plural noun as its lexical head)."""
    global __CONCEPTUAL_CATEGORIES__
    if '__CONCEPTUAL_CATEGORIES__' not in globals():
        __CONCEPTUAL_CATEGORIES__ = util.load_or_create_cache('dbpedia_categories_conceptual', _compute_conceptual_categories)

    return category in __CONCEPTUAL_CATEGORIES__


def _compute_conceptual_categories() -> set:
    util.get_logger().info('CACHE: Computing conceptual categories')
    return {cat for cat in cat_store.get_categories() if nlp_util.get_head_lemmas(cat_store.get_label(cat))}
