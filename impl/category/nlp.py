import impl.util.nlp as nlp_util
import impl.category.store as cat_store
import util
import inflection
from spacy.tokens import Doc

SINGULARIZED_CATEGORIES_CACHE_ID = 'categories_singularized'


def parse_category(category: str) -> Doc:
    """Return the category name as parsed Doc."""
    label = cat_store.get_label(category)
    return nlp_util.parse(label)


def singularize(category: str) -> str:
    """Return a singularized version of the category where only the lexical head of the category is singularized."""
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
    if '__SINGULARIZED_CATEGORIES__' in globals():
        util.update_cache(SINGULARIZED_CATEGORIES_CACHE_ID, __SINGULARIZED_CATEGORIES__)
