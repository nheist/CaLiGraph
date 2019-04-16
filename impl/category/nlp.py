import impl.util.nlp as nlp_util
import impl.category.store as cat_store
import util
import inflection
from spacy.tokens import Doc, Span

SINGULARIZED_CATEGORIES_CACHE_ID = 'categories_singularized'


def is_conceptual(category: str) -> bool:
    global __CONCEPTUAL_CATEGORIES__
    if '__CONCEPTUAL_CATEGORIES__' not in globals():
        __CONCEPTUAL_CATEGORIES__ = util.load_or_create_cache('categories_conceptual', _compute_conceptual_categories)

    return category in __CONCEPTUAL_CATEGORIES__


def _compute_conceptual_categories() -> set:
    util.get_logger().info('Computing conceptual categories..')
    # TODO: implement check with category sets to find remaining categories (eg. the 5k albums that we are missing out)
    conceptual_categories = set()
    for cat in cat_store.get_all_cats():
        doc = _tag_lexical_head(_parse_category(cat))
        if any(word.tag_ == 'NNS' and word.ent_type_ == 'LH' for word in doc):
            conceptual_categories.add(cat)
    return conceptual_categories


def _tag_lexical_head(doc: Doc) -> Doc:
    chunk_words = {w for chunk in doc.noun_chunks for w in chunk}
    lexhead_start = None
    for chunk in doc.noun_chunks:
        elem = chunk.root
        if elem.text.istitle():
            continue
        if elem.tag_ not in ['NN', 'NNS']:
            continue
        if len(doc) > elem.i + 1 and doc[elem.i+1].text in ['(', ')', '–']:
            continue
        if len(doc) > elem.i + 2 and doc[elem.i+1].text == 'and' and doc[elem.i+2] in chunk_words:
            lexhead_start = lexhead_start if lexhead_start is not None else chunk.start
            continue
#            if doc[elem.i+2].pos_ == 'NOUN':
#                elem = doc[elem.i:elem.i+3]
#            elif len(doc) > elem.i + 3 and doc[elem.i+2].pos_ == 'ADJ' and doc[elem.i+3].pos_ == 'NOUN':
#                elem = doc[elem.i:elem.i+4]
        doc.ents = [Span(doc, lexhead_start or chunk.start, chunk.end, label=doc.vocab.strings['LH'])]
    return doc


def _parse_category(category: str) -> Doc:
    label = cat_store.get_label(category)
    split_label = label.split(' ')
    if len(split_label) > 1 and not (label[1].isupper() or split_label[1][0].isupper()):
        label = label[0].lower() + label[1:]
    return nlp_util.parse(label)


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
