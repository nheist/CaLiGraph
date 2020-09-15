from pytest_check import check_func
import impl.dbpedia.heuristics as dbp_heur
from impl.dbpedia.util import NAMESPACE_DBP_ONTOLOGY as DBO


def test_disjoint_types():
    # Person
    _assert_disjoint('Person', 'Activity')
    _assert_disjoint('Person', 'Event')
    _assert_disjoint('Person', 'Place')
    _assert_disjoint('Person', 'Organisation')
    _assert_disjoint('Person', 'Work')
    _assert_not_disjoint('Person', 'Artist')
    _assert_not_disjoint('Person', 'Agent')
    # Artist
    _assert_not_disjoint('Artist', 'Architect')
    _assert_not_disjoint('Artist', 'Astronaut')
    _assert_not_disjoint('Artist', 'Coach')
    _assert_not_disjoint('Artist', 'Monarch')
    # Place
    _assert_disjoint('Place', 'Award')
    _assert_disjoint('Place', 'EthnicGroup')
    _assert_disjoint('Place', 'Food')
    _assert_disjoint('Place', 'MeanOfTransportation')
    _assert_disjoint('Place', 'Work')
    _assert_not_disjoint('Place', 'Organisation')
    _assert_not_disjoint('Place', 'Company')
    _assert_not_disjoint('Place', 'University')
    _assert_not_disjoint('Place', 'School')
    # Organisation
    _assert_disjoint('Organisation', 'Activity')
    _assert_disjoint('Organisation', 'Event')
    _assert_disjoint('Organisation', 'Award')
    _assert_disjoint('Organisation', 'Device')
    _assert_disjoint('Organisation', 'Work')
    _assert_disjoint('Organisation', 'UnitOfWork')
    _assert_disjoint('Organisation', 'Species')
    _assert_not_disjoint('Organisation', 'Hospital')
    _assert_not_disjoint('Organisation', 'Casino')
    _assert_not_disjoint('Organisation', 'Museum')
    _assert_not_disjoint('Organisation', 'Restaurant')
    # Event
    _assert_disjoint('Event', 'Activity')
    _assert_disjoint('Event', 'Agent')
    _assert_disjoint('Event', 'Device')
    _assert_disjoint('Event', 'Food')
    _assert_disjoint('Event', 'MeanOfTransportation')
    _assert_disjoint('Event', 'Species')
    _assert_disjoint('Event', 'Work')
    # Species
    _assert_disjoint('Species', 'Agent')
    _assert_disjoint('Species', 'Event')
    _assert_disjoint('Species', 'Activity')
    _assert_disjoint('Species', 'Disease')
    _assert_disjoint('Species', 'Place')
    _assert_disjoint('Species', 'Work')
    _assert_disjoint('Species', 'Award')
    # Work
    _assert_disjoint('Work', 'Agent')
    _assert_disjoint('Work', 'Activity')
    _assert_disjoint('Work', 'AnatomicalStructure')
    _assert_disjoint('Work', 'Food')
    _assert_disjoint('Work', 'TimePeriod')
    # Album
    _assert_disjoint('Album', 'Agent')
    _assert_disjoint('Album', 'Artwork')
    _assert_disjoint('Album', 'Event')
    _assert_disjoint('Album', 'Language')
    _assert_disjoint('Album', 'Musical')
    # Film
    _assert_disjoint('Film', 'WrittenWork')
    _assert_disjoint('Film', 'Artwork')
    _assert_disjoint('Film', 'Agent')
    _assert_disjoint('Film', 'Place')


@check_func
def _assert_disjoint(type1: str, type2: str):
    assert DBO + type2 in dbp_heur.get_disjoint_types(DBO + type1), f'{type1} should be disjoint with {type2}'


@check_func
def _assert_not_disjoint(type1: str, type2: str):
    assert DBO + type2 not in dbp_heur.get_disjoint_types(DBO + type1), f'{type1} should not be disjoint with {type2}'
