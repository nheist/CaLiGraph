import util
import impl.util.nlp as nlp_util
import impl.category.store as cat_store
import impl.category.nlp as cat_nlp
import impl.category.category_set as cat_set
from spacy.tokens import Doc


def is_conceptual_category(category: str) -> bool:
    """Return true, if the category is conceptual (i.e. has a plural noun as its lexical head)."""
    global __CONCEPTUAL_CATEGORIES__
    if '__CONCEPTUAL_CATEGORIES__' not in globals():
        __CONCEPTUAL_CATEGORIES__ = util.load_or_create_cache('dbpedia_categories_conceptual', _compute_conceptual_categories)

    return category in __CONCEPTUAL_CATEGORIES__


def _compute_conceptual_categories() -> set:
    util.get_logger().info('CACHE: Computing conceptual categories')
    return {cat for cat, doc in _tag_lexical_heads(cat_store.get_categories()).items() if _has_plural_lexical_head(doc)}


def _tag_lexical_heads(categories: set) -> dict:
    """Identify lexical head by basic tagging and additionally by using information from category sets."""
    # basic lexical head tagging
    category_docs = {cat: nlp_util.tag_lexical_head(cat_nlp.parse_category(cat)) for cat in categories}
    plural_lexhead_cats = {cat for cat, doc in category_docs.items() if _has_plural_lexical_head(doc)}

    # enhanced lexical head tagging with category sets (yields 2822 additional conceptual categories)
    for category_set in cat_set.get_category_sets():

        set_cats = category_set.nodes
        pattern_words = set(category_set.pattern[0] + category_set.pattern[1])

        plural_lexhead_set_cats = {c for c in set_cats if c in plural_lexhead_cats}
        plural_lexhead_ratio = len(plural_lexhead_set_cats) / len(set_cats)
        if .5 <= plural_lexhead_ratio < 1:
            plural_lexheads = [{w.text for w in category_docs[cat] if w.ent_type_ == 'LH'} for cat in plural_lexhead_set_cats]
            if all(pattern_words.issubset(plural_lexhead) for plural_lexhead in plural_lexheads):
                other_cats = {c for c in set_cats if c not in plural_lexhead_set_cats}
                for c in other_cats:
                    category_docs[c] = nlp_util.tag_lexical_head(cat_nlp.parse_category(c), valid_words=pattern_words)

    return category_docs


def _has_plural_lexical_head(doc: Doc) -> bool:
    return any(chunk.root.tag_ == 'NNS' and chunk.root.ent_type_ == 'LH' for chunk in doc.noun_chunks)


