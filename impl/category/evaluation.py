from .graph import CategoryGraph
import impl.dbpedia.store as dbp_store
import impl.category.store as cat_store
from collections import defaultdict
from typing import Tuple
import itertools
import pandas as pd
import numpy as np
import util


def test_metrics(graph: CategoryGraph):
    util.get_logger().debug('Evaluation of parameters for dbp-type extraction..')

    columns = ['apply_impure_type_filtering', 'exclude_untyped_resources', 'prefer_resource_types', 'apply_type_depth_smoothing', 'resource_type_ratio', 'child_type_ratio', 'precision', 'recall', 'F1']
    data = list(itertools.product([True, False], [True, False], [True, False], [True, False], np.linspace(.1, 1, 10), np.linspace(.1, 1, 10), [0.0], [0.0], [0.0]))
    df = pd.DataFrame(data=data, columns=columns)

    for idx, row in enumerate(df.itertuples()):
        util.set_config('category.dbp_types.apply_impure_type_filtering', row.apply_impure_type_filtering)
        util.set_config('category.dbp_types.exclude_untyped_resources', row.exclude_untyped_resources)
        util.set_config('category.dbp_types.prefer_resource_types', row.prefer_resource_types)
        util.set_config('category.dbp_types.apply_type_depth_smoothing', row.apply_type_depth_smoothing)
        util.set_config('category.dbp_types.resource_type_ratio', row.resource_type_ratio)
        util.set_config('category.dbp_types.child_type_ratio', row.child_type_ratio)

        eval_graph = graph.copy()
        eval_graph.assign_dbp_types()
        precision, recall, f1 = _get_metrics(eval_graph)

        df.at[row.Index, 'precision'] = precision
        df.at[row.Index, 'recall'] = recall
        df.at[row.Index, 'F1'] = f1
        util.get_logger().debug(f'RUN {idx}/{len(df.index)}: P={precision*100:.2f} R={recall*100:.2f} F1={f1*100:.2f}')

    df.to_csv('catgraph-dbptype_evaluation.csv')
    util.get_logger().debug('Finished evaluation of parameters for dbp-type extraction.')


def _get_metrics(graph: CategoryGraph):
    tp, fp, fn, tn = 0, 0, 0, 0
    goldstandard = pd.read_csv(util.get_data_file('files.evaluation.catgraph_dbptypes'))

    for cat in goldstandard['cat'].values:
        types_possible = set(goldstandard[goldstandard['cat'] == cat]['dbp_type'].values)
        types_actual = set(goldstandard[(goldstandard['cat'] == cat) & (goldstandard['label'] == 1)]['dbp_type'].values)
        types_graph = graph.dbp_types(cat)

        tp += len(types_actual.intersection(types_graph))
        fp += len(types_graph.difference(types_actual))
        fn += len(types_actual.difference(types_graph))
        tn += len(types_possible.difference(types_actual | types_graph))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def test_custom_metrics(graph: CategoryGraph):
    columns = ['exclude_untyped_resources', 'resource_type_ratio', 'child_type_ratio', 'correctness', 'accordance', 'recall', 'coverage']
    data = list(itertools.product([True, False], np.linspace(0.1, 1, 10), np.linspace(0.1, 1, 10), [0.0], [0.0], [0.0], [0.0]))
    df = pd.DataFrame(data=data, columns=columns)

    for row in df.itertuples():
        util.set_config('category.dbp_types.exclude_untyped_resources', row.exclude_untyped_resources)
        util.set_config('category.dbp_types.resource_type_ratio', row.resource_type_ratio)
        util.set_config('category.dbp_types.child_type_ratio', row.child_type_ratio)

        eval_graph = graph.copy()
        eval_graph.assign_dbp_types()
        correctness, accordance, recall, coverage = _get_custom_metrics(eval_graph)

        df.at[row.Index, 'correctness'] = correctness
        df.at[row.Index, 'accordance'] = accordance
        df.at[row.Index, 'recall'] = recall
        df.at[row.Index, 'coverage'] = coverage

    df.to_csv('catgraph_evaluation.csv')


def _get_custom_metrics(graph: CategoryGraph) -> Tuple[float, float, float, float]:
    graph_resource_types = _get_graph_resource_type_mapping(graph)
    return _get_correctness(graph_resource_types), _get_accordance(graph_resource_types), _get_recall(graph_resource_types), _get_coverage(graph)


def _get_correctness(graph_resource_types: dict) -> float:
    """ 1 - Proportion of resources with disjoint types assigned """
    resources = dbp_store.get_resources()
    disjoint_type_resources = set()
    for r in resources:
        possible_disjoint_types = {dt for t in dbp_store.get_types(r) for dt in dbp_store.get_disjoint_types(t)}
        if possible_disjoint_types.intersection(graph_resource_types[r]):
            disjoint_type_resources.add(r)
    return 1 - (len(disjoint_type_resources) / len(resources))


def _get_accordance(graph_resource_types: dict) -> float:
    """ 1 - Proportion of resources with assigned types that are not subtypes of their currently most specific type """
    resources = dbp_store.get_resources()
    nonaccording_type_resources = set()
    for r in resources:
        independent_resource_types = dbp_store.get_independent_types(dbp_store.get_types(r))
        subtypes_of_most_specific_types = {tst for it in independent_resource_types for tst in dbp_store.get_transitive_subtypes(it)}
        possible_according_types = dbp_store.get_transitive_types(r) | subtypes_of_most_specific_types
        if graph_resource_types[r].difference(possible_according_types):
            nonaccording_type_resources.add(r)
    return 1 - (len(nonaccording_type_resources) / len(resources))


def _get_recall(graph_resource_types: dict) -> float:
    """ Proportion of correctly found types """
    found_types = 0
    overall_types = 0
    for r in dbp_store.get_resources():
        dbp_types = dbp_store.get_transitive_types(r)
        found_types += len(dbp_types.intersection(graph_resource_types[r]))
        overall_types += len(dbp_types)
    return found_types / overall_types


def _get_coverage(graph: CategoryGraph) -> float:
    return len({cat for cat in graph.nodes if graph.dbp_types(cat)}) / len(graph.nodes)


def _get_graph_resource_type_mapping(graph: CategoryGraph) -> dict:
    resources = dbp_store.get_resources()
    category_type_mapping = defaultdict(set, {cat: graph.dbp_types(cat) or set() for cat in graph.nodes})
    resource_category_mapping = cat_store.get_resource_to_cats_mapping()
    return {res: {t for cat in resource_category_mapping[res] for t in category_type_mapping[cat]} for res in resources}
