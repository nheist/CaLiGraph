import impl.util.nlp as nlp_util
import impl.category.store as cat_store
import util
import inflection
import spacy

SINGULARIZED_CATEGORIES_CACHE_ID = 'categories_singularized'


def is_conceptual(category: str) -> bool:
    global __CONCEPTUAL_CATEGORIES__
    if '__CONCEPTUAL_CATEGORIES__' not in globals():
        __CONCEPTUAL_CATEGORIES__ = util.load_or_create_cache('categories_conceptual', _compute_conceptual_categories)

    return category in __CONCEPTUAL_CATEGORIES__


def _compute_conceptual_categories() -> set:
    # TODO: implement check with category sets to find remaining categories (eg. the 5k albums that we are missing out)
    conceptual_caegories = set()
    for cat in cat_store.get_all_cats():
        lexhead = get_lexical_head(cat)
        if lexhead:
            main_token = lexhead if type(lexhead) == spacy.tokens.Token else lexhead[-1]
            if main_token.tag_ == 'NNS':
                conceptual_caegories.add(cat)
    return conceptual_caegories


def get_lexical_head(category: str):
    label = cat_store.get_label(category)
    split_label = label.split(' ')
    if len(split_label) > 1 and not (label[1].isupper() or split_label[1][0].isupper()):
        label = label[0].lower() + label[1:]
    doc = nlp_util.parse(label)
    for chunk in doc.noun_chunks:
        elem = chunk.root
        if (elem.text.istitle() and elem != doc[0]) or elem.text == '–':
            continue
        if len(doc) > elem.i + 1 and doc[elem.i+1].text == ')':
            continue
        if len(doc) > elem.i + 2 and doc[elem.i+1].text == 'and':
            if doc[elem.i+2].pos_ == 'NOUN':
                elem = doc[elem.i:elem.i+3]
            elif len(doc) > elem.i + 3 and doc[elem.i+2].pos_ == 'ADJ' and doc[elem.i+3].pos_ == 'NOUN':
                elem = doc[elem.i:elem.i+4]
        return elem
    return None


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
    if '__SINGULARIZED_CATEGORIES__' in globals():
        util.update_cache(SINGULARIZED_CATEGORIES_CACHE_ID, __SINGULARIZED_CATEGORIES__)
