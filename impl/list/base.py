import pandas as pd
import util
from . import parser as list_parser
from . import features as list_features
from . import extract as list_extract
from impl.list.graph import ListGraph
from impl.caligraph.graph import CaLiGraph
import impl.dbpedia.store as dbp_store
import impl.dbpedia.heuristics as dbp_heur
from collections import defaultdict


# LIST HIERARCHY

def get_base_listgraph() -> ListGraph:
    global __BASE_LISTGRAPH__
    if '__BASE_LISTGRAPH__' not in globals():
        initializer = lambda: ListGraph.create_from_dbpedia().append_unconnected()
        __BASE_LISTGRAPH__ = util.load_or_create_cache('listgraph_base', initializer)
    return __BASE_LISTGRAPH__


def get_wikitaxonomy_listgraph() -> ListGraph:
    global __WIKITAXONOMY_LISTGRAPH__
    if '__WIKITAXONOMY_LISTGRAPH__' not in globals():
        initializer = lambda: get_base_listgraph().remove_unrelated_edges()
        __WIKITAXONOMY_LISTGRAPH__ = util.load_or_create_cache('listgraph_wikitaxonomy', initializer)
    return __WIKITAXONOMY_LISTGRAPH__


def get_cyclefree_wikitaxonomy_listgraph() -> ListGraph:
    global __CYCLEFREE_WIKITAXONOMY_LISTGRAPH__
    if '__CYCLEFREE_WIKITAXONOMY_LISTGRAPH__' not in globals():
        initializer = lambda: get_wikitaxonomy_listgraph().resolve_cycles().append_unconnected()
        __CYCLEFREE_WIKITAXONOMY_LISTGRAPH__ = util.load_or_create_cache('listgraph_cyclefree', initializer)
    return __CYCLEFREE_WIKITAXONOMY_LISTGRAPH__


def get_merged_listgraph() -> ListGraph:
    global __MERGED_LISTGRAPH__
    if '__MERGED_LISTGRAPH__' not in globals():
        initializer = lambda: get_cyclefree_wikitaxonomy_listgraph().merge_nodes()
        __MERGED_LISTGRAPH__ = util.load_or_create_cache('listgraph_merged', initializer)
    return __MERGED_LISTGRAPH__


# LIST ENTITIES

def get_filtered_listpage_entities(graph: CaLiGraph, listpage: str) -> set:
    global __FILTERED_LISTPAGE_ENTITIES__
    if '__FILTERED_LISTPAGE_ENTITIES__' not in globals():
        initializer = lambda: _filter_listpage_entities(graph)
        __FILTERED_LISTPAGE_ENTITIES__ = defaultdict(set, util.load_or_create_cache('dbpedia_filtered_listpage_entities', initializer))
    return __FILTERED_LISTPAGE_ENTITIES__[listpage]


def _filter_listpage_entities(graph: CaLiGraph) -> dict:
    get_listpage_entities('')  # make sure that listpage entities are initialised
    filtered_entities = {}
    for lp, entities in __LISTPAGE_ENTITIES__.items():
        caligraph_nodes = graph.get_nodes_for_part(lp)
        lp_types = {t for n in caligraph_nodes for types in graph.get_dbpedia_types(n) for t in types}
        disjoint_types = {dt for t in lp_types for dt in dbp_heur.get_disjoint_types(t)}
        valid_entities = {e for e in entities if not disjoint_types.intersection(dbp_store.get_transitive_types(e))}
        filtered_entities[lp] = valid_entities  # TODO: check if it makes sense to discard all lp entities at a certain threshold
    return filtered_entities


def get_listpage_entities(listpage: str) -> set:
    global __LISTPAGE_ENTITIES__
    if '__LISTPAGE_ENTITIES__' not in globals():
        __LISTPAGE_ENTITIES__ = defaultdict(set, util.load_or_create_cache('dbpedia_listpage_entities', _extract_listpage_entities))
    return __LISTPAGE_ENTITIES__[listpage]


def _extract_listpage_entities():
    enum_features = get_enum_listpage_entity_features()
    enum_entities = list_extract.extract_enum_entities(enum_features)

    table_features = get_table_listpage_entity_features()
    table_entities = list_extract.extract_table_entities(table_features)

    return {lp: enum_entities[lp] | table_entities[lp] for lp in (set(enum_entities) | set(table_entities))}


def get_enum_listpage_entity_features() -> pd.DataFrame:
    global __ENUM_LISTPAGE_ENTITY_FEATURES__
    if '__ENUM_LISTPAGE_ENTITY_FEATURES__' not in globals():
        __ENUM_LISTPAGE_ENTITY_FEATURES__ = util.load_or_create_cache('dbpedia_listpage_enum_features', lambda: _compute_listpage_entity_features(list_parser.LIST_TYPE_ENUM))
    return __ENUM_LISTPAGE_ENTITY_FEATURES__


def get_table_listpage_entity_features() -> pd.DataFrame:
    global __TABLE_LISTPAGE_ENTITY_FEATURES__
    if '__TABLE_LISTPAGE_ENTITY_FEATURES__' not in globals():
        __TABLE_LISTPAGE_ENTITY_FEATURES__ = util.load_or_create_cache('dbpedia_listpage_table_features', lambda: _compute_listpage_entity_features(list_parser.LIST_TYPE_TABLE))
    return __TABLE_LISTPAGE_ENTITY_FEATURES__


def _compute_listpage_entity_features(list_type: str) -> pd.DataFrame:
    util.get_logger().info(f'List-Entities: Computing entity features for {list_type}..')

    entity_features = []
    parsed_listpages = list_parser.get_parsed_listpages()
    for idx, (lp, lp_data) in enumerate(parsed_listpages.items()):
        if idx % 1000 == 0:
            util.get_logger().debug(f'List-Entities: Processed {idx} of {len(parsed_listpages)} listpages.')

        if lp_data['type'] != list_type:
            continue
        if list_type == list_parser.LIST_TYPE_ENUM:
            entity_features.extend(list_features.make_enum_entity_features(lp_data))
        elif list_type == list_parser.LIST_TYPE_TABLE:
            entity_features.extend(list_features.make_table_entity_features(lp_data))
    entity_features = pd.DataFrame(data=entity_features)

    entity_features.to_csv('table_entity_backup_preencode.csv', sep=';')  # TODO: REMOVE!

    entity_features = list_features.onehotencode_feature(entity_features, '_section_name')
    entity_features = list_features.onehotencode_feature(entity_features, '_column_name')

    entity_features.to_csv('table_entity_backup_prelabel.csv', sep=';')  # TODO: REMOVE!

    #entity_features.to_hdf('table_entity_backup.h5', key='df', mode='w')  # TODO: REMOVE!
    #entity_features = pd.read_csv('table_entity_backup.csv', sep=';', index_col=0)
    util.get_logger().info('List-Entities: Assigning entity labels..')
    list_features.assign_entity_labels(entity_features)

    util.get_logger().info('List-Entities: Finished extracting entity features.')
    return entity_features
