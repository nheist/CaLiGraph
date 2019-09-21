import spacy
from spacy.tokens import Doc, Span
import hashlib
import util
import re
import inflection

SPACY_CACHE_ID = 'spacy_docs'

# initialization
parser = spacy.load('en_core_web_lg')
# manually initializing stop words as they are missing in lg corpus
for word in parser.Defaults.stop_words:
    lex = parser.vocab[word]
    lex.is_stop = True


def without_stopwords(text: str) -> set:
    """Return the lemmatized versions of all non-stop-words in `text`."""
    text = _remove_parentheses_content(text.replace('-', ' '))
    return {word.lemma_ for word in parse(text) if not word.is_stop}


def get_canonical_name(text: str, disable_normalization=True) -> str:
    text = remove_by_phrase(parse(text, disable_normalization=disable_normalization), return_doc=False)  # remove by-phrase
    text = re.sub(r'\s+\([^()]+[-–][^()]+\)$', '', text)  # remove trailing parentheses with number or letter ranges, e.g. 'Interstate roads (1-10)'
    text = re.sub(r'\s+\([A-Z]\)$', '', text)  # remove trailing parentheses with single letter, e.g. 'Interstate roads (Y)'
    text = re.sub(r'\s*[-:,–]\s*[A-Z][a-z]*\s?[-–]\s?[A-Z][a-z]*$', '', text)  # remove trailing alphabetical ranges, e.g. 'Drugs: Sp-Sub'
    text = re.sub(r'\s*[-:–]\s*([A-Z],\s*)*[A-Z]$', '', text)  # remove trailing alphabetical splits, e.g. 'Football clubs in Sweden - Z' or '.. - X, Y, Z'
    text = re.sub(r'\s+([A-Z],\s*)+[A-Z]$', '', text)  # remove trailing alphabetical splits without indicator, e.g. 'Fellows of the Royal Society A, B, C'
    return _regularize_whitespaces(text)


def remove_by_phrase(doc: Doc, return_doc=True):
    """Remove the 'by'-phrase at the end of a category or listpage, e.g. 'People by country' -> 'People'"""
    by_indices = [w.i for w in doc if w.text == 'by']
    if len(by_indices) == 0:
        return doc if return_doc else doc.text
    last_by_index = by_indices[-1]
    if last_by_index == 0 or last_by_index == len(doc) - 1:
        return doc if return_doc else doc.text
    word_before_by = doc[last_by_index-1]
    word_after_by = doc[last_by_index+1]
    if word_after_by.text[0].isupper() or word_after_by.tag_ == 'NNS' or word_after_by.text in ['a', 'an', 'the'] or word_before_by.tag_ == 'VBN':
        return doc if return_doc else doc.text

    result = doc[:last_by_index].text.strip()
    if return_doc:
        return parse(result, disable_normalization=True, skip_cache=True)
    return result


def get_head_lemmas(doc: Doc) -> set:
    """Return the lemmatized version of the lexical head of `doc`."""
    doc = tag_lexical_head(doc)
    head_lemmas = set()
    for idx, w in enumerate(doc):
        if w.ent_type_ != 'LH' or w.tag_ != 'NNS':
            continue
        if idx + 1 < len(doc) and doc[idx+1].ent_type_ == 'LH' and doc[idx+1].text != 'and':
            continue
        head_lemmas.add(w.lemma_)
    return head_lemmas


def tag_lexical_head(doc: Doc) -> Doc:
    """Return `doc` where the lexical head is tagged as the entity 'LH'."""

    if len(doc) == 0:
        return doc

    # ensure that numbers are also regarded as nouns if being stand-alone
    if doc[0].tag_ == 'CD' and (len(doc) < 2 or not doc[1].tag_.startswith('NN')):
        doc.ents = [Span(doc, 0, 1, label=doc.vocab.strings['LH'])]
        return doc

    chunk_words = {w for chunk in doc.noun_chunks for w in chunk}
    lexhead_start = None
    for chunk in doc.noun_chunks:
        # find the lexical head by looking for plural nouns (and ignore things like parentheses, conjunctions, ..)
        elem = chunk.root
        if elem.text.istitle() or elem.tag_ not in ['NN', 'NNS']:
            continue
        if len(doc) > elem.i + 1 and doc[elem.i+1].text in ['(', ')', '–'] and doc[-1].text != ')':
            continue
        if (len(doc) > elem.i + 1 and doc[elem.i + 1].tag_ in ['NN', 'NNS']) or (len(doc) > elem.i + 2 and doc[elem.i+1].text == 'and' and doc[elem.i+2] in chunk_words):
            lexhead_start = lexhead_start if lexhead_start is not None else chunk.start
            continue
        lexhead_start = lexhead_start if lexhead_start is not None else chunk.start
        doc.ents = [Span(doc, lexhead_start, chunk.end, label=doc.vocab.strings['LH'])]
        break
    return doc


def singularize_phrase(doc: Doc) -> str:
    doc = tag_lexical_head(doc)
    result = doc.text
    for idx, w in enumerate(doc):
        if w.ent_type_ == 'LH' and (len(doc) <= idx+1 or doc[idx+1].text == 'and' or doc[idx+1].ent_type_ != 'LH'):
            result = result.replace(w.text, inflection.singularize(w.text))
            if len(doc) > idx+1 and doc[idx+1].text == 'and':
                result = result.replace('and', 'or')
    return result


def parse(text: str, disable_normalization=False, skip_cache=False) -> Doc:
    if not disable_normalization and text:
        split_text = text.split(' ')
        if len(split_text) == 1 or (len(split_text) > 1 and not (text[1].isupper() or split_text[1].istitle())):
            text = text[0].lower() + text[1:] if len(text) > 1 else text[0].lower()

    if skip_cache:
        return parser(text)

    global __NLP_CACHE__, __NLP_CACHE_CHANGED__
    if '__NLP_CACHE__' not in globals():
        __NLP_CACHE__ = util.load_or_create_cache(SPACY_CACHE_ID, lambda: dict())
        __NLP_CACHE_CHANGED__ = False

    text_hash = hashlib.md5(text.encode('utf-8')).digest()
    if text_hash in __NLP_CACHE__:
        return __NLP_CACHE__[text_hash]

    parsed_text = parser(text)
    __NLP_CACHE__[text_hash] = parsed_text
    __NLP_CACHE_CHANGED__ = True
    return parsed_text


def _remove_parentheses_content(text: str) -> str:
    without_parentheses = re.sub(r'\([^()]*\)', '', text)
    return _regularize_whitespaces(without_parentheses)


def _regularize_whitespaces(text: str) -> str:
    result = re.sub(r'\s+', ' ', text).strip()
    result = result[:-1] if result.endswith(',') else result
    return result


def persist_cache():
    global __NLP_CACHE_CHANGED__
    if '__NLP_CACHE_CHANGED__' in globals() and __NLP_CACHE_CHANGED__:
        util.update_cache(SPACY_CACHE_ID, __NLP_CACHE__)
        __NLP_CACHE_CHANGED__ = False
