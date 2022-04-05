from pytest_check import check_func
from impl.caligraph.ontology import ClgOntologyStore
from impl.caligraph.entity import ClgEntityStore
from impl.dbpedia.resource import DbpResourceStore
from impl.dbpedia.category import DbpCategoryStore


def test_classes():
    _is_not_in_graph(f'Demographic')  # TODO: Fix! (e.g. with better non-set category identification)


def test_class_hierarchy():
    _is_parent_of(f'Air_force', f'Disbanded_air_force')
    _is_no_parent_of(f'Air_force', f'Air_force_personnel')

    _is_parent_of(f'Bodybuilder', f'Fitness_or_figure_competitor')  # TODO: Fix!
    _is_no_parent_of(f'Fitness_or_figure_competition', f'Fitness_or_figure_competitor')  # TODO: Fix!

    _is_parent_of(f'Bodybuilding_competition', f'Female_bodybuilding_competition')
    _is_no_parent_of(f'Person', f'Female_bodybuilding_competition')

    _is_parent_of(f'University_and_college_person', f'Alumni')  # TODO: Fix!
    _is_no_parent_of(f'Honduran_person', f'Alumni')
    _is_no_parent_of(f'Papua_New_Guinean_person', f'Alumni')

    _is_no_parent_of(f'Japanese_sportsperson', f'Association_football_person')
    _is_parent_of(f'Football_person', f'Association_football_person')

    _is_ancestor_of(f'Person', f'Sportswoman')  # TODO: Fix!
    _is_parent_of(f'Woman', f'Sportswoman')  # TODO: Fix!

    _is_no_parent_of(f'Place', f'Etymology')  # TODO: Fix!


def test_class_resources():
    _is_resource_of(f'What_Separates_Me_from_You', f'Album_produced_by_Chad_Gilbert')
    _is_no_resource_of(f'What_Separates_Me_from_You', f'Song_written_by_Chad_Gilbert')  # TODO: Fixed?


def test_node_parts():
    _is_part_of(f'Organisation', f'Organization')


def test_singularization():
    _is_part_of(f'Category:Engineering_societies_by_country', f'Engineering_society')
    _is_part_of(f'List_of_engineering_societies', f'Engineering_society')

    _is_part_of(f'List_of_sportswomen', f'Sportswoman')

    _is_no_part_of(f'List_of_caves', f'Cafe')


def test_by_phrase_removal():
    _is_part_of(f'List_of_countries_by_national_capital_and_largest_cities', f'Country')  # TODO: Fix!
    _is_part_of(f'List_of_countries_by_vehicles_per_capita', f'Country')  # TODO: Fix!
    _is_part_of(f'List_of_countries_by_Nobel_laureates_per_capita', f'Country')  # TODO: Fix!
    _is_part_of(f'List_of_countries_by_number_of_households', f'Country')  # TODO: Fix!
    _is_part_of(f'List_of_countries_by_health_expenditure_covered_by_government', f'Country')
    _is_part_of(f'Category:Lists_of_countries_by_per_capita_values', f'Country')  # TODO: Fix!
    _is_part_of(f'Category:Lists_of_countries_by_GDP_per_capita', f'Country')
    _is_part_of(f'Category:Vehicles_by_brand_controlled_by_Volkswagen_Group', f'Vehicle')
    _is_part_of(f'Category:African_films_by_genre_by_country', f'African_film')
    _is_part_of(f'Category:Alumni_by_university_or_college_in_Honduras', f'Alumni_in_Honduras')
    _is_no_part_of(f'Category:Books_by_bell_hooks', f'American_book')  # TODO: Fixed?
    _is_no_part_of(f'Category:Novels_by_DBC_Pierre', f'Novel')  # TODO: Fixed?

    _is_in_graph(f'Work_by_Roy_Lichtenstein')
    _is_in_graph(f'Work_by_L._J._Smith')
    _is_in_graph(f'Film_produced_by_Harry_Saltzman')
    _is_in_graph(f'20th-century_execution_by_Sweden')
    _is_in_graph(f'United_States_Article_I_federal_judge_appointed_by_Jimmy_Carter')
    _is_in_graph(f'Bundesliga_club_eliminated_from_the_DFB-Pokal_by_amateur_sides')
    _is_in_graph(f'Opera_by_Krenek')
    _is_in_graph(f'Work_by_Presidents_of_the_United_States')
    _is_in_graph(f'Electronica_album_by_South_African_artists')
    _is_in_graph(f'Person_killed_by_live_burial_by_Nazi_Germany')
    _is_in_graph(f'Killing_by_law_enforcement_officers_by_countries')
    _is_in_graph(f'Pop_album_by_Scottish_artists')
    _is_in_graph(f'Song_recorded_by_ABBA')
    _is_in_graph(f'Program_broadcast_by_France_2')
    _is_in_graph(f'International_cycle_race_hosted_by_the_United_Kingdom')


@check_func
def _is_parent_of(parent: str, child: str):
    clgo = ClgOntologyStore.instance()
    parent = clgo.get_class_by_name(parent)
    child = clgo.get_class_by_name(child)
    assert child in clgo.subtypes(parent), f'{parent} should be parent of {child}'


@check_func
def _is_ancestor_of(ancestor: str, child: str):
    clgo = ClgOntologyStore.instance()
    ancestor = clgo.get_class_by_name(ancestor)
    child = clgo.get_class_by_name(child)
    assert ancestor in clgo.get_transitive_supertypes(child), f'{ancestor} should be ancestor of {child}'


@check_func
def _is_no_parent_of(parent: str, child: str):
    clgo = ClgOntologyStore.instance()
    parent = clgo.get_class_by_name(parent)
    child = clgo.get_class_by_name(child)
    assert child not in clgo.subtypes(parent), f'{parent} should not be parent of {child}'


@check_func
def _is_in_graph(node: str):
    clgo = ClgOntologyStore.instance()
    assert clgo.has_class_with_name(node), f'{node} should be in the graph'


@check_func
def _is_not_in_graph(node: str):
    clgo = ClgOntologyStore.instance()
    assert not clgo.has_class_with_name(node), f'{node} should not be in the graph'


@check_func
def _is_part_of(part: str, node: str):
    clgo = ClgOntologyStore.instance()
    dbr = DbpResourceStore.instance()
    dbc = DbpCategoryStore.instance()

    assert clgo.has_class_with_name(node)

    clg_type = clgo.get_class_by_name(node)
    if dbr.has_resource_with_name(part):
        part = dbr.get_resource_by_name(part)
    elif dbc.has_category_with_name(part):
        part = dbc.get_category_by_name(part)
    assert part in clg_type.get_associated_dbp_resources(), f'{part} should be part of {node}'


@check_func
def _is_no_part_of(part: str, node: str):
    clgo = ClgOntologyStore.instance()
    dbr = DbpResourceStore.instance()
    dbc = DbpCategoryStore.instance()

    if clgo.has_class_with_name(node):
        clg_type = clgo.get_class_by_name(node)
        if dbr.has_resource_with_name(part):
            part = dbr.get_resource_by_name(part)
        elif dbc.has_category_with_name(part):
            part = dbc.get_category_by_name(part)
        assert part not in clg_type.get_associated_dbp_resources(), f'{part} should not be part of {node}'


@check_func
def _is_resource_of(res: str, node: str):
    clgo = ClgOntologyStore.instance()
    clge = ClgEntityStore.instance()
    assert clgo.has_class_with_name(node)
    clg_type = clgo.get_class_by_name(node)
    clg_ent = clge.get_entity_by_name(res)
    assert clg_type in clg_ent.get_types(), f'{res} should be contained in {node}'


@check_func
def _is_no_resource_of(res: str, node: str):
    clgo = ClgOntologyStore.instance()
    clge = ClgEntityStore.instance()
    if clgo.has_class_with_name(node):
        clg_type = clgo.get_class_by_name(node)
        clg_ent = clge.get_entity_by_name(res)
        assert clg_type not in clg_ent.get_types(), f'{res} should not be contained in {node}'
