import pandas as pd


def create_relation_frame(data_dict: dict) -> pd.SparseDataFrame:
    index = data_dict.keys()
    columns = list({t for types in data_dict.values() for t in types})
    data = [[c in data_dict[i] for c in columns] for i in index]
    return pd.SparseDataFrame(data=data, index=index, columns=columns, dtype=bool)


def create_value_frame(data_dict: dict, value_name: str) -> pd.DataFrame:
    index = data_dict.keys()
    data = [data_dict[i] for i in index]
    return pd.DataFrame(data={value_name: data}, index=index)
