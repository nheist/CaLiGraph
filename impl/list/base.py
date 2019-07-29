import pandas as pd
import util
from . import parser as list_parser
from . import features as list_features
from impl.list.graph import ListGraph


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
        initializer = lambda: get_cyclefree_wikitaxonomy_listgraph().merge_nodes().resolve_cycles()
        __MERGED_LISTGRAPH__ = util.load_or_create_cache('listgraph_merged', initializer)
    return __MERGED_LISTGRAPH__


# LIST ENTITIES

def get_listpage_entity_features() -> pd.DataFrame:
    global __LISTPAGE_ENTITY_FEATURES__
    if '__LISTPAGE_ENTITY_FEATURES__' not in globals():
        __LISTPAGE_ENTITY_FEATURES__ = util.load_or_create_cache('dbpedia_listpage_features', _compute_listpage_entity_features)
    return __LISTPAGE_ENTITY_FEATURES__


def _compute_listpage_entity_features() -> pd.DataFrame:
    util.get_logger().info('List-Entities: Computing entity features..')

    entities = None
    parsed_listpages = list_parser.get_parsed_listpages()
    progress = 0
    for lp, lp_data in parsed_listpages.items():
        progress += 1
        if progress % 1000 == 0:
            util.get_logger().debug(f'List-Entities: Extracted features for {progress} of {len(parsed_listpages)}')
        if lp_data['type'] != list_parser.LIST_TYPE_ENUM:
            continue

        lp_entities = list_features.make_entity_features(lp_data)
        entities = entities.append(lp_entities, ignore_index=True) if entities is not None else lp_entities

    entities = list_features.with_section_name_features(entities)

    util.get_logger().info('List-Entities: Assigning entity labels..')
    list_features.assign_entity_labels(get_merged_listgraph(), entities)

    util.get_logger().info('List-Entities: Finished extracting entity features.')
    return entities
