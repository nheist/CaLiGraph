"""Functionality to retrieve cached versions of the list graph and entities in several stages."""

import pandas as pd
import util
import impl.dbpedia.page_features as page_features
import impl.dbpedia.util as dbp_util
import impl.list.store as list_store
import impl.list.entity_labels as list_entity_labels
import impl.list.extract as list_extract
import impl.list.nlp as list_nlp
from impl.list.graph import ListGraph
from collections import defaultdict
import multiprocessing as mp
from impl import wikipedia
from tqdm import tqdm


# LIST HIERARCHY

def get_base_listgraph() -> ListGraph:
    """Retrieve basic list graph without any modifications."""
    global __BASE_LISTGRAPH__
    if '__BASE_LISTGRAPH__' not in globals():
        initializer = lambda: ListGraph.create_from_dbpedia().append_unconnected()
        __BASE_LISTGRAPH__ = util.load_or_create_cache('listgraph_base', initializer)
    return __BASE_LISTGRAPH__


def get_wikitaxonomy_listgraph() -> ListGraph:
    """Retrieve list graph with filtered edges."""
    global __WIKITAXONOMY_LISTGRAPH__
    if '__WIKITAXONOMY_LISTGRAPH__' not in globals():
        initializer = lambda: get_base_listgraph().remove_unrelated_edges()
        __WIKITAXONOMY_LISTGRAPH__ = util.load_or_create_cache('listgraph_wikitaxonomy', initializer)
    return __WIKITAXONOMY_LISTGRAPH__


def get_cyclefree_wikitaxonomy_listgraph() -> ListGraph:
    """Retrieve list graph with filtered edges and resolved cycles."""
    global __CYCLEFREE_WIKITAXONOMY_LISTGRAPH__
    if '__CYCLEFREE_WIKITAXONOMY_LISTGRAPH__' not in globals():
        initializer = lambda: get_wikitaxonomy_listgraph().append_unconnected()
        __CYCLEFREE_WIKITAXONOMY_LISTGRAPH__ = util.load_or_create_cache('listgraph_cyclefree', initializer)
    return __CYCLEFREE_WIKITAXONOMY_LISTGRAPH__


def get_merged_listgraph() -> ListGraph:
    """Retrieve list graph with filtered edges, resolved cycles, and merged lists."""
    global __MERGED_LISTGRAPH__
    if '__MERGED_LISTGRAPH__' not in globals():
        initializer = lambda: get_cyclefree_wikitaxonomy_listgraph().merge_nodes().remove_leaf_listcategories().remove_transitive_edges()
        __MERGED_LISTGRAPH__ = util.load_or_create_cache('listgraph_merged', initializer)
    return __MERGED_LISTGRAPH__


# LIST ENTITIES

def get_listpage_entities(graph, listpage_uri: str) -> dict:
    """Retrieve the extracted entities of a given list page."""
    global __LISTPAGE_ENTITIES__
    if '__LISTPAGE_ENTITIES__' not in globals():
        __LISTPAGE_ENTITIES__ = defaultdict(dict, util.load_or_create_cache('dbpedia_listpage_entities', lambda: _extract_listpage_entities(graph)))
    return __LISTPAGE_ENTITIES__[dbp_util.resource2name(listpage_uri)]


def _extract_listpage_entities(graph):
    """Extract and return entities for all list pages."""
    util.get_logger().info('LIST/BASE: Extracting new entities from list pages..')

    util.get_logger().info('LIST/BASE: Extracting enum entities..')
    enum_features = get_enum_listpage_entity_features(graph)
    enum_entities = list_extract.extract_enum_entities(enum_features)

    util.get_logger().info('LIST/BASE: Extracting table entities..')
    table_features = get_table_listpage_entity_features(graph)
    table_entities = list_extract.extract_table_entities(table_features)

    listpage_entities = {}
    for lp in set(enum_entities) | set(table_entities):
        entities = defaultdict(set, enum_entities[lp])
        for ent, labels in table_entities[lp].items():
            entities[ent].update(labels)
        listpage_entities[lp] = entities
    return listpage_entities


def get_enum_listpage_entity_features(graph) -> pd.DataFrame:
    """Extract entities from enumeration list pages."""
    global __ENUM_LISTPAGE_ENTITY_FEATURES__
    if '__ENUM_LISTPAGE_ENTITY_FEATURES__' not in globals():
        __ENUM_LISTPAGE_ENTITY_FEATURES__ = util.load_or_create_cache('dbpedia_listpage_enum_features', lambda: _compute_listpage_entity_features(graph, wikipedia.ARTICLE_TYPE_ENUM))
    return __ENUM_LISTPAGE_ENTITY_FEATURES__


def get_table_listpage_entity_features(graph) -> pd.DataFrame:
    """Extract entities from table list pages."""
    global __TABLE_LISTPAGE_ENTITY_FEATURES__
    if '__TABLE_LISTPAGE_ENTITY_FEATURES__' not in globals():
        __TABLE_LISTPAGE_ENTITY_FEATURES__ = util.load_or_create_cache('dbpedia_listpage_table_features', lambda: _compute_listpage_entity_features(graph, wikipedia.ARTICLE_TYPE_TABLE))
    return __TABLE_LISTPAGE_ENTITY_FEATURES__


def _compute_listpage_entity_features(graph, list_type: str) -> pd.DataFrame:
    """Compute entity features depending on the layout type of a list page."""
    util.get_logger().info(f'LIST/BASE: Computing entity features for {list_type}..')

    # make sure that the spacy list parser is initialized before using it in multiple processes
    list_nlp._initialise_parser()

    parsed_listpages = list_store.get_parsed_listpages(list_type)
    feature_func = page_features.make_enum_entity_features if list_type == wikipedia.ARTICLE_TYPE_ENUM else page_features.make_table_entity_features
    with mp.Pool(processes=round(util.get_config('max_cpus') / 2)) as pool:
        entity_features = [x for examples in tqdm(pool.imap_unordered(feature_func, parsed_listpages.items(), chunksize=1000), total=len(parsed_listpages)) for x in examples]
    column_names = page_features.get_enum_feature_names() if list_type == wikipedia.ARTICLE_TYPE_ENUM else page_features.get_table_feature_names()
    entity_features = pd.DataFrame(data=entity_features, columns=column_names)

    util.get_logger().info('LIST/BASE: Assigning entity labels..')
    list_entity_labels.assign_entity_labels(graph, entity_features)

    return entity_features
