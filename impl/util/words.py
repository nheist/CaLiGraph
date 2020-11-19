import inflection
from nltk.corpus import words, wordnet


def is_english_plural_word(word: str) -> bool:
    """Check whether the given word is the plural form of an English word."""
    singular_word = inflection.singularize(word)
    if word == singular_word:
        return False
    return singular_word in get_english_words() or singular_word.lower() in get_english_words()


def get_english_words() -> set:
    """Return all English words in the dictionary of nltk."""
    global __WORDS__
    if '__WORDS__' not in globals():
        __WORDS__ = set(words.words())
    return __WORDS__


def get_synonyms(word: str) -> set:
    """Returns all synonyms of a word from WordNet."""
    return {lm.name() for syn in wordnet.synsets(word) for lm in syn.lemmas()}
