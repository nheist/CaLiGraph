import inflection
import nltk
nltk.download('words')
from nltk.corpus import words


def is_english_plural_word(word: str) -> bool:
    singular_word = inflection.singularize(word)
    return singular_word in get_english_words() or singular_word.lower() in get_english_words()


def get_english_words() -> set:
    global __WORDS__
    if '__WORDS__' not in globals():
        __WORDS__ = set(words.words())
    return __WORDS__
