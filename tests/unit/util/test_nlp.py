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
    assert nlp_util.remove_by_phrase(nlp_util.parse(with_by_phrase, skip_cache=True)).lower() == without_by_phrase.lower(),\
        f'{with_by_phrase} should be converted to {without_by_phrase}'


def test_get_head_lemmas():
    pass


def test_singularize_phrase():
    pass
