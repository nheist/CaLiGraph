import caligraph.util.nlp as nlp_util
import caligraph.category.store as cat_store
import util

CONCEPTUAL_CATEGORIES_CACHE_ID = 'conceptual-categories'


def is_conceptual(category: str) -> bool:
    if '__CONCEPTUAL_CATEGORIES__' not in globals():
        global __CONCEPTUAL_CATEGORIES__
        __CONCEPTUAL_CATEGORIES__ = util.load_or_create_cache(CONCEPTUAL_CATEGORIES_CACHE_ID, lambda: dict())

    if category not in __CONCEPTUAL_CATEGORIES__:
        label = cat_store.get_label(category)
        label_chunks = list(nlp_util.parse(label).noun_chunks)
        __CONCEPTUAL_CATEGORIES__[category] = (label_chunks[0][-1].tag_ == 'NNS') if len(label_chunks) > 0 else False

    return __CONCEPTUAL_CATEGORIES__[category]


def singularize(category: str) -> str:
    pass


def persist_cache():
    nlp_util.persist_cache()
    util.update_cache(CONCEPTUAL_CATEGORIES_CACHE_ID, __CONCEPTUAL_CATEGORIES__)
