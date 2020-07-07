"""Functionality to retrieve cached versions of the list graph and entities in several stages."""

import pandas as pd
import math
from itertools import islice
import util
import impl.list.store as list_store
import impl.list.features as list_features
import impl.list.extract as list_extract
import impl.list.nlp as list_nlp
from impl.list.graph import ListGraph
from collections import defaultdict
import multiprocessing as mp


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
        initializer = lambda: get_wikitaxonomy_listgraph().resolve_cycles().append_unconnected()
        __CYCLEFREE_WIKITAXONOMY_LISTGRAPH__ = util.load_or_create_cache('listgraph_cyclefree', initializer)
    return __CYCLEFREE_WIKITAXONOMY_LISTGRAPH__


def get_merged_listgraph() -> ListGraph:
    """Retrieve list graph with filtered edges, resolved cycles, and merged lists."""
    global __MERGED_LISTGRAPH__
    if '__MERGED_LISTGRAPH__' not in globals():
        initializer = lambda: get_cyclefree_wikitaxonomy_listgraph().merge_nodes().remove_transitive_edges()
        __MERGED_LISTGRAPH__ = util.load_or_create_cache('listgraph_merged', initializer)
    return __MERGED_LISTGRAPH__


# LIST ENTITIES

def get_listpage_entities(graph, listpage: str) -> set:
    """Retrieve the extracted entities of a given list page."""
    global __LISTPAGE_ENTITIES__
    if '__LISTPAGE_ENTITIES__' not in globals():
        __LISTPAGE_ENTITIES__ = defaultdict(set, util.load_or_create_cache('dbpedia_listpage_entities', lambda: _extract_listpage_entities(graph)))
    return __LISTPAGE_ENTITIES__[listpage]


def _extract_listpage_entities(graph):
    """Extract and return entities for all list pages."""
    util.get_logger().info('LIST/BASE: Extracting new entities from list pages..')

    util.get_logger().info('LIST/BASE: Extracting enum entities..')
    enum_features = get_enum_listpage_entity_features(graph)
    enum_entities = list_extract.extract_enum_entities(enum_features)

    util.get_logger().info('LIST/BASE: Extracting table entities..')
    table_features = get_table_listpage_entity_features(graph)
    table_entities = list_extract.extract_table_entities(table_features)

    listpage_entities = {lp: enum_entities[lp] | table_entities[lp] for lp in (set(enum_entities) | set(table_entities))}
    return listpage_entities


def get_enum_listpage_entity_features(graph) -> pd.DataFrame:
    """Extract entities from enumeration list pages."""
    global __ENUM_LISTPAGE_ENTITY_FEATURES__
    if '__ENUM_LISTPAGE_ENTITY_FEATURES__' not in globals():
        __ENUM_LISTPAGE_ENTITY_FEATURES__ = util.load_or_create_cache('dbpedia_listpage_enum_features', lambda: _compute_listpage_entity_features(graph, list_store.LIST_TYPE_ENUM))
    return __ENUM_LISTPAGE_ENTITY_FEATURES__


def get_table_listpage_entity_features(graph) -> pd.DataFrame:
    """Extract entities from table list pages."""
    global __TABLE_LISTPAGE_ENTITY_FEATURES__
    if '__TABLE_LISTPAGE_ENTITY_FEATURES__' not in globals():
        __TABLE_LISTPAGE_ENTITY_FEATURES__ = util.load_or_create_cache('dbpedia_listpage_table_features', lambda: _compute_listpage_entity_features(graph, list_store.LIST_TYPE_TABLE))
    return __TABLE_LISTPAGE_ENTITY_FEATURES__


def _compute_listpage_entity_features(graph, list_type: str) -> pd.DataFrame:
    """Compute entity features depending on the layout type of a list page."""
    util.get_logger().info(f'LIST/BASE: Computing entity features for {list_type}..')

    # make sure that the spacy list parser is initialized before using it in multiple processes
    list_nlp._initialise_parser()

    parsed_listpages = list_store.get_parsed_listpages(list_type)
    feature_func = list_features.make_enum_entity_features if list_type == list_store.LIST_TYPE_ENUM else list_features.make_table_entity_features
    number_of_processes = round(util.get_config('max_cpus') / 2)
    with mp.Pool(processes=number_of_processes) as pool:
        params = [(lp_chunk, feature_func) for lp_chunk in _chunk_dict(parsed_listpages, number_of_processes)]
        entity_features = [example for examples in pool.starmap(_run_feature_extraction_for_listpages, params) for example in examples]
    entity_features = pd.DataFrame(data=entity_features)

    # TODO: remove this!
    if list_type == list_store.LIST_TYPE_TABLE:
        entity_features.to_csv('table_intermediate_save.csv', sep=';')
    # /TODO: remove this!

    # one-hot-encode name features
    util.get_logger().info('LIST/BASE: One-hot encoding features..')
    entity_features = list_features.onehotencode_feature(entity_features, '_section_name')
    if '_column_name' in entity_features.columns:
        entity_features = list_features.onehotencode_feature(entity_features, '_column_name')

    util.get_logger().info('LIST/BASE: Assigning entity labels..')
    list_features.assign_entity_labels(graph, entity_features)

    return entity_features


def _chunk_dict(dict_to_chunk: dict, number_of_chunks: int):
    chunk_size = math.ceil(len(dict_to_chunk) / number_of_chunks)
    it = iter(dict_to_chunk)
    for _ in range(0, len(dict_to_chunk), chunk_size):
        yield {k: dict_to_chunk[k] for k in islice(it, chunk_size)}


def _run_feature_extraction_for_listpages(parsed_listpages: dict, feature_func) -> list:
    entity_features = []
    for idx, (lp, lp_data) in enumerate(parsed_listpages.items()):
        if idx % 1000 == 0:
            util.get_logger().debug(f'LIST/BASE ({mp.current_process().name}): Processed {idx} of {len(parsed_listpages)} listpages.')
        entity_features.extend(feature_func(lp, lp_data))
    return entity_features
