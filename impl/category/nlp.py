import impl.util.nlp as nlp_util
import impl.category.store as cat_store
import util
import inflection

CONCEPTUAL_CATEGORIES_CACHE_ID = 'categories_conceptual'
SINGULARIZED_CATEGORIES_CACHE_ID = 'categories_singularized'


def is_conceptual(category: str) -> bool:
    global __CONCEPTUAL_CATEGORIES__
    if '__CONCEPTUAL_CATEGORIES__' not in globals():
        __CONCEPTUAL_CATEGORIES__ = util.load_or_create_cache(CONCEPTUAL_CATEGORIES_CACHE_ID, lambda: dict())

    if category not in __CONCEPTUAL_CATEGORIES__:
        label = cat_store.get_label(category)
        if len(label) > 1 and not label[1].isupper():
            label = label[0].lower() + label[1:]
        label_chunks = list(nlp_util.parse(label).noun_chunks)
        __CONCEPTUAL_CATEGORIES__[category] = label_chunks and label.startswith(label_chunks[0].text) and label_chunks[0][-1].tag_ == 'NNS'

    return __CONCEPTUAL_CATEGORIES__[category]


def singularize(category: str) -> str:
    global __SINGULARIZED_CATEGORIES__
    if '__SINGULARIZED_CATEGORIES__' not in globals():
        __SINGULARIZED_CATEGORIES__ = util.load_or_create_cache(SINGULARIZED_CATEGORIES_CACHE_ID, lambda: dict())

    if category not in __SINGULARIZED_CATEGORIES__:
        label = cat_store.get_label(category)
        label_chunks = list(nlp_util.parse(label).noun_chunks)
        chunk_to_singularize = label_chunks[0].text.split(' ')
        if len(chunk_to_singularize) > 1 and chunk_to_singularize[-2] == 'and':
            chunk_part = '_'.join(chunk_to_singularize[-3:])
            singularized_chunk_part = '_'.join([inflection.singularize(chunk_to_singularize[-3]), 'or', inflection.singularize(chunk_to_singularize[-1])])
        else:
            chunk_part = chunk_to_singularize[-1]
            singularized_chunk_part = inflection.singularize(chunk_part)
        __SINGULARIZED_CATEGORIES__[category] = category.replace(chunk_part, singularized_chunk_part)

    return __SINGULARIZED_CATEGORIES__[category]


def persist_cache():
    nlp_util.persist_cache()
    if '__CONCEPTUAL_CATEGORIES__' in globals():
        util.update_cache(CONCEPTUAL_CATEGORIES_CACHE_ID, __CONCEPTUAL_CATEGORIES__)
    if '__SINGULARIZED_CATEGORIES__' in globals():
        util.update_cache(SINGULARIZED_CATEGORIES_CACHE_ID, __SINGULARIZED_CATEGORIES__)
