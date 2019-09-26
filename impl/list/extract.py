import pandas as pd
from xgboost import XGBClassifier
from collections import defaultdict


def extract_enum_entities(df: pd.DataFrame) -> dict:
    # TODO: optimise with grid-search
    estimator = XGBClassifier(random_state=42, colsample_bytree=.8, max_depth=5, n_estimators=200, scale_pos_weight=.25)
    return _extract_entities(df, estimator)


def extract_table_entities(df: pd.DataFrame) -> dict:
    # TODO: optimise with grid-search
    estimator = XGBClassifier(random_state=42, colsample_bytree=.8, max_depth=5, n_estimators=200, scale_pos_weight=.25)
    return _extract_entities(df, estimator)


def _extract_entities(df: pd.DataFrame, estimator) -> dict:
    df_true = df[df['label'] == 1].copy()
    df_new = df[df['label'] == -1].copy()

    # prepare data
    df = df.drop(columns=[c for c in df.columns.values if c.startswith('_')])  # remove id columns
    df = pd.get_dummies(df)
    train, candidates = df[df['label'] != -1], df[df['label'] == -1].drop(columns='label')
    X, y = train.drop(columns='label'), train['label']

    # predict
    estimator.fit(X, y)
    df_new['label'] = estimator.predict(candidates)

    # extract true entities
    list_entities = defaultdict(set)
    for idx, row in pd.concat([df_true, df_new[df_new['label'] == 1]]).iterrows():
        list_entities[row['_listpage_uri']].add(row['_entity_uri'])
    return list_entities
