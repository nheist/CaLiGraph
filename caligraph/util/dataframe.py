import pandas as pd
import caligraph.util.rdf as rdf_util
from collections import defaultdict


def create_relation_frame_from_rdf(filepaths: list, predicate: str) -> pd.DataFrame:
    data_dict = defaultdict(list)
    for fp in filepaths:
        for triple in rdf_util.parse_triples_from_file(fp):
            if triple.pred == predicate:
                data_dict[triple.sub].append(triple.obj)
    return create_relation_frame(data_dict)


def create_value_frame_from_rdf(filepaths: list, predicate: str) -> pd.DataFrame:
    data_dict = {}
    for fp in filepaths:
        for triple in rdf_util.parse_triples_from_file(fp):
            if triple.pred == predicate:
                data_dict[triple.sub] = triple.obj
    return create_value_frame(data_dict, predicate)


def create_relation_frame(data_dict: dict) -> pd.DataFrame:
    index = data_dict.keys()
    columns = list({t for types in data_dict.values() for t in types})
    data = [[c in data_dict[i] for c in columns] for i in index]
    return pd.SparseDataFrame(data=data, index=index, columns=columns, dtype=bool)


def create_value_frame(data_dict: dict, value_name: str) -> pd.DataFrame:
    index = data_dict.keys()
    data = {value_name: [data_dict[i] for i in index]}
    return pd.DataFrame(data=data, index=index)
