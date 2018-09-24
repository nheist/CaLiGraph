import caligraph.util.nlp as nlp_util
import caligraph.category.store as cat_store
import util
import inflection

CONCEPTUAL_CATEGORIES_CACHE_ID = 'conceptual_categories'
SINGULARIZED_CATEGORIES_CACHE_ID = 'singularized_categories'


def is_conceptual(category: str) -> bool:
    if '__CONCEPTUAL_CATEGORIES__' not in globals():
        global __CONCEPTUAL_CATEGORIES__
        __CONCEPTUAL_CATEGORIES__ = util.load_or_create_cache(CONCEPTUAL_CATEGORIES_CACHE_ID, lambda: dict())

    if category not in __CONCEPTUAL_CATEGORIES__:
        label = cat_store.get_label(category)
        label_chunks = list(nlp_util.parse(label).noun_chunks)
        __CONCEPTUAL_CATEGORIES__[category] = label_chunks and label.startswith(label_chunks[0].text) and label_chunks[0][-1].tag_ == 'NNS'

    return __CONCEPTUAL_CATEGORIES__[category]


def singularize(category: str) -> str:
    if '__SINGULARIZED_CATEGORIES__' not in globals():
        global __SINGULARIZED_CATEGORIES__
        __SINGULARIZED_CATEGORIES__ = util.load_or_create_cache(SINGULARIZED_CATEGORIES_CACHE_ID, lambda: dict())

    if category not in __SINGULARIZED_CATEGORIES__:
        label = cat_store.get_label(category)
        label_chunks = list(nlp_util.parse(label).noun_chunks)
        chunk_to_singularize = label_chunks[0]
        if len(chunk_to_singularize) > 1 and chunk_to_singularize[-2] == 'and':
            chunk_part = '_'.join(chunk_to_singularize[-3:])
            singularized_chunk_part = '_'.join([inflection.singularize(chunk_to_singularize[-3]), 'or', inflection.singularize(chunk_to_singularize[-1])])
        else:
            chunk_part = chunk_to_singularize[-1]
            singularized_chunk_part = inflection.singularize(chunk_to_singularize[-1])
        __SINGULARIZED_CATEGORIES__[category] = category.replace(chunk_part, singularized_chunk_part)

    return __SINGULARIZED_CATEGORIES__[category]


def persist_cache():
    nlp_util.persist_cache()
    if '__CONCEPTUAL_CATEGORIES__' in globals():
        util.update_cache(CONCEPTUAL_CATEGORIES_CACHE_ID, __CONCEPTUAL_CATEGORIES__)
    if '__SINGULARIZED_CATEGORIES__' in globals():
        util.update_cache(SINGULARIZED_CATEGORIES_CACHE_ID, __SINGULARIZED_CATEGORIES__)
