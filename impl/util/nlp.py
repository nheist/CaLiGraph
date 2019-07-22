import spacy
from spacy.tokens import Doc, Span
import hashlib
import util

SPACY_CACHE_ID = 'spacy_docs'

# initialization
parser = spacy.load('en_core_web_lg')
# manually initializing stop words as they are missing in lg corpus
for word in parser.Defaults.stop_words:
    lex = parser.vocab[word]
    lex.is_stop = True


def filter_important_words(text: str) -> set:
    """Return the lemmatized versions of all non-stop-words in `text`."""
    return {word.lemma_ for word in remove_by_phrase(parse(text)) if not word.is_stop}


def remove_by_phrase_from_text(text: str) -> str:
    return remove_by_phrase(parse(text, disable_normalization=True, skip_cache=True)).text


def remove_by_phrase(doc: Doc) -> Doc:
    """Remove the 'by'-phrase at the end of a category or listpage, e.g. 'People by country' -> 'People'"""
    by_indices = [w.i for w in doc if w.text == 'by']
    if len(by_indices) == 0:
        return doc
    last_by_index = by_indices[-1]
    if last_by_index == 0 or last_by_index == len(doc) - 1:
        return doc
    word_after_by = doc[last_by_index+1]
    if word_after_by.text.istitle() or word_after_by.text == 'the' or word_after_by.tag_ == 'NNS':
        return doc
    return doc[:last_by_index].as_doc()


def get_head_lemmas(doc: Doc) -> set:
    """Return the lemmatized version of the lexical head of `doc`."""
    doc = tag_lexical_head(doc)
    return {w.lemma_ for w in doc if w.tag_ == 'NNS' and w.ent_type_ == 'LH'}


def tag_lexical_head(doc: Doc, valid_words=None) -> Doc:
    """Return `doc` where the lexical head is tagged as the entity 'LH'."""
    chunk_words = {w for chunk in doc.noun_chunks for w in chunk}
    lexhead_start = None
    for chunk in doc.noun_chunks:
        if valid_words and all(w not in chunk.text for w in valid_words):
            continue

        # find the lexical head by looking for plural nouns (and ignore things like parentheses, conjunctions, ..)
        elem = chunk.root
        if elem.text.istitle() or elem.tag_ not in ['NN', 'NNS']:
            continue
        if len(doc) > elem.i + 1 and doc[elem.i+1].text in ['(', ')', '–']:
            continue
        if (len(doc) > elem.i + 1 and doc[elem.i + 1].tag_ in ['NN', 'NNS']) or (len(doc) > elem.i + 2 and doc[elem.i+1].text == 'and' and doc[elem.i+2] in chunk_words):
            lexhead_start = lexhead_start if lexhead_start is not None else chunk.start
            continue
        lexhead_start = lexhead_start if lexhead_start is not None else chunk.start
        doc.ents = [Span(doc, lexhead_start, chunk.end, label=doc.vocab.strings['LH'])]
        break
    return doc


def parse(text: str, disable_normalization=False, skip_cache=False) -> Doc:
    if not disable_normalization:
        split_text = text.split(' ')
        if len(split_text) == 1 or (len(split_text) > 1 and not (text[1].isupper() or split_text[1].istitle())):
            text = text[0].lower() + text[1:]

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


def persist_cache():
    global __NLP_CACHE_CHANGED__
    if '__NLP_CACHE_CHANGED__' in globals() and __NLP_CACHE_CHANGED__:
        util.update_cache(SPACY_CACHE_ID, __NLP_CACHE__)
        __NLP_CACHE_CHANGED__ = False
