from pytest_check import check_func
import impl.util.nlp as nlp_util


def test_without_stopwords():
    pass


def test_remove_parentheses_content():
    pass


def test_get_canonical_name():
    pass


def test_remove_by_phrase():
    _without_by_phrase('Work by Roy Lichtenstein', 'Work by Roy Lichtenstein')
    _without_by_phrase('Work by L. J. Smith', 'Work by L. J. Smith')
    _without_by_phrase('Song recorded by ABBA', 'Song recorded by ABBA')
    _without_by_phrase('Alumni by university or college in Honduras', 'Alumni in Honduras')
    _without_by_phrase('Countries by GDP per capita', 'Countries')


@check_func
def _without_by_phrase(with_by_phrase: str, without_by_phrase: str):
    with_removed_by_phrase = nlp_util.remove_by_phrase(with_by_phrase).text
    assert with_removed_by_phrase.lower() == without_by_phrase.lower(), f'{with_by_phrase} should be converted to {without_by_phrase}'


def test_has_plural_lexhead_subjects():
    _text_has_plural_lexhead_subjects('Reptiles')
    _text_has_plural_lexhead_subjects('Princesses')
    _text_has_plural_lexhead_subjects('African musical instruments')

    _text_has_no_plural_lexhead_subjects('London')


@check_func
def _text_has_plural_lexhead_subjects(text: str):
    assert nlp_util.has_plural_lexhead_subjects(text)


@check_func
def _text_has_no_plural_lexhead_subjects(text: str):
    assert not nlp_util.has_plural_lexhead_subjects(text)


def test_singularize_phrase():
    pass
