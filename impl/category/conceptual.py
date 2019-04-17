import util
import impl.category.store as cat_store
import impl.category.nlp as cat_nlp
import impl.category.category_set as cat_set
from spacy.tokens import Doc, Span


def is_conceptual_category(category: str) -> bool:
    global __CONCEPTUAL_CATEGORIES__
    if '__CONCEPTUAL_CATEGORIES__' not in globals():
        __CONCEPTUAL_CATEGORIES__ = util.load_or_create_cache('dbpedia_categories_conceptual', _compute_conceptual_categories)

    return category in __CONCEPTUAL_CATEGORIES__


def _compute_conceptual_categories() -> set:
    util.get_logger().info('Computing conceptual categories..')
    return {cat for cat, doc in _tag_lexical_heads(cat_store.get_all_cats()).items() if _has_plural_lexical_head(doc)}


def _tag_lexical_heads(categories) -> dict:
    # basic lexical head tagging
    category_docs = {cat: _tag_lexical_head(cat_nlp.parse_category(cat)) for cat in categories}
    plural_lexhead_cats = {cat for cat, doc in category_docs.items() if _has_plural_lexical_head(doc)}

    # enhanced lexical head tagging with category sets
    for idx, category_set in enumerate(cat_set.get_category_sets()):
        if idx % 1000 == 0:
            util.get_logger().debug(f'Processed {idx}/{len(cat_set.get_category_sets())} category sets for LH tagging.')

        set_cats = category_set['categories']
        pattern_words = category_set['pattern'][0] + category_set['pattern'][1]

        plural_lexhead_set_cats = {c for c in set_cats if c in plural_lexhead_cats}
        plural_lexhead_ratio = len(plural_lexhead_set_cats) / len(set_cats)
        if .5 <= plural_lexhead_ratio < 1:
            plural_lexheads = [{w.text for w in category_docs[cat] if w.ent_type_ == 'LH'} for cat in plural_lexhead_set_cats]
            if all(pattern_words.issubset(plural_lexhead) for plural_lexhead in plural_lexheads):
                other_cats = {c for c in set_cats if c not in plural_lexhead_set_cats}
                for c in other_cats:
                    pre_lexhead = [w.text for w in category_docs[c] if w.ent_type_ == 'LH']
                    category_docs[c] = _tag_lexical_head(cat_nlp.parse_category(c), valid_words=pattern_words)
                    post_lexhead = [w.text for w in category_docs[c] if w.ent_type_ == 'LH']
                    util.get_logger().debug(f'Changed lexhead for category {c} from {" ".join(pre_lexhead)} to {" ".join(post_lexhead)}')

    return category_docs


def _tag_lexical_head(doc: Doc, valid_words=None) -> Doc:
    chunk_words = {w for chunk in doc.noun_chunks for w in chunk}
    lexhead_start = None
    for chunk in doc.noun_chunks:
        if valid_words and all(w not in chunk.text for w in valid_words):
            continue

        elem = chunk.root
        if elem.text.istitle() or elem.tag_ not in ['NN', 'NNS']:
            continue
        if len(doc) > elem.i + 1 and doc[elem.i+1].text in ['(', ')', '–']:
            continue
        if len(doc) > elem.i + 2 and doc[elem.i+1].text == 'and' and doc[elem.i+2] in chunk_words:
            lexhead_start = lexhead_start if lexhead_start is not None else chunk.start
            continue
        lexhead_start = lexhead_start if lexhead_start is not None else chunk.start
        doc.ents = [Span(doc, lexhead_start, chunk.end, label=doc.vocab.strings['LH'])]
        break
    return doc


def _has_plural_lexical_head(doc: Doc) -> bool:
    return any(chunk.root.tag_ == 'NNS' and chunk.root.ent_type_ == 'LH' for chunk in doc.noun_chunks)


