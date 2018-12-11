import spacy
from spacy.tokens import Doc
import hashlib
import util

SPACY_CACHE_ID = 'spacy_docs'

# initialization
parser = spacy.load('en_core_web_lg')


def parse(text: str, skip_cache=False) -> Doc:
    global __NLP_CACHE__, __NLP_CACHE_CHANGED__
    if '__NLP_CACHE__' not in globals():
        __NLP_CACHE__ = util.load_or_create_cache(SPACY_CACHE_ID, lambda: dict())
        __NLP_CACHE_CHANGED__ = False

    text_hash = hashlib.md5(text.encode('utf-8')).digest()
    if text_hash in __NLP_CACHE__:
        return __NLP_CACHE__[text_hash]

    parsed_text = parser(text)
    if not skip_cache:
        __NLP_CACHE__[text_hash] = parsed_text
        __NLP_CACHE_CHANGED__ = True
    return parsed_text


def persist_cache():
    global __NLP_CACHE_CHANGED__
    if '__NLP_CACHE_CHANGED__' in globals() and __NLP_CACHE_CHANGED__:
        util.update_cache(SPACY_CACHE_ID, __NLP_CACHE__)
        __NLP_CACHE_CHANGED__ = False
