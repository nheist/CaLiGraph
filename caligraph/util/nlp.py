import spacy
from spacy.tokens import Doc
import hashlib
import util

SPACY_CACHE_ID = 'spacy_docs'

# initialization
parser = spacy.load('en_core_web_lg')
cache = util.load_or_create_cache(SPACY_CACHE_ID, lambda: dict())


def parse(text: str, skip_cache=False) -> Doc:
    text_hash = hashlib.md5(text)
    if text_hash in cache:
        return cache[text_hash]

    parsed_text = parser(text)
    if not skip_cache:
        cache[text_hash] = parsed_text
    return parsed_text


def persist_cache():
    util.update_cache(SPACY_CACHE_ID, cache)
