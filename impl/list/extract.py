import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier
from collections import defaultdict


def extract_enum_entities(df: pd.DataFrame) -> dict:
    df_true = df[df['label'] == 1].copy()
    df_new = df[df['label'] == -1].copy()

    # prepare data
    df = df.drop(columns=[c for c in df.columns.values if c.startswith('_')])  # remove id columns
    df = pd.get_dummies(df)
    train, candidates = df[df['label'] != -1], df[df['label'] == -1].drop(columns='label')
    X, y = train.drop(columns='label'), train['label']

    # define model
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(k=100)),
        ('classification', XGBClassifier(random_state=42, colsample_bytree=.6, max_depth=15, n_estimators=300)),
    ])

    # predict
    pipeline.fit(X, y)
    df_new['label'] = pipeline.predict(candidates)

    list_entities = defaultdict(set)
    for idx, row in pd.concat([df_true, df_new[df_new['label'] == 1]]).iterrows():
        list_entities[row['_listpage_uri']].add(row['_entity_uri'])
    return list_entities
