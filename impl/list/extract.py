"""Training and application of machine learning models for subject entity extraction."""

import math
import random
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from collections import defaultdict, namedtuple
import multiprocessing as mp
import util


def extract_enum_entities(df: pd.DataFrame) -> dict:
    """Return entities extracted from enumeration list pages with XG-Boost-based collective extraction."""
    config = {
        'base_estimator': XGBClassifier(predictor='cpu_predictor', colsample_bytree=.8, max_depth=5, n_estimators=400, scale_pos_weight=.25),
        'sampling_functions': [_sample_by_entity_position],
        'selection': {
            'model': XGBRegressor(predictor='cpu_predictor'),
            'n_candidates': 3,
            'min_score': .4,
        }
    }
    return _extract_entities(df, config)


def extract_table_entities(df: pd.DataFrame) -> dict:
    """Return entities extracted from table list pages with XG-Boost-based collective extraction."""
    base_estimator = Pipeline([
        ('feature_selection', SelectKBest(k=100)),
        ('classification', XGBClassifier(predictor='cpu_predictor', colsample_bytree=.8, max_depth=5, n_estimators=200, scale_pos_weight=.25)),
    ])
    config = {
        'base_estimator': base_estimator,
        'sampling_functions': [_sample_by_column],
        'selection': {
            'model': XGBRegressor(predictor='cpu_predictor'),
            'n_candidates': 3,
            'min_score': .4,
        }
    }
    return _extract_entities(df, config)


def _extract_entities(df: pd.DataFrame, config: dict) -> dict:
    # -- assign mentions the probability of being subject entities (using individual classification) --
    # prepare params
    estimator = config['base_estimator']
    meta_columns = [c for c in df.columns.values if c.startswith('_')] + ['label']
    df_data = pd.get_dummies(df.drop(columns=meta_columns))
    df = pd.merge(df_data, df[meta_columns], left_index=True, right_index=True, how='inner')
    # get training data
    df_train = df[df['label'] != -1]
    # split into two training sets
    gss = GroupShuffleSplit(n_splits=1, train_size=.5, random_state=42)
    train1_idxs, train2_idxs = next(gss.split(df_train, y=df_train['label'], groups=df_train['_listpage_uri']))
    # train model for proba and apply
    df_train1 = df_train.iloc[train1_idxs]
    estimator.fit(pd.get_dummies(df_train1.drop(columns=meta_columns)), df_train1['label'])
    true_label_idx = np.where(estimator.classes_ == 1)[0][0]
    df['proba'] = [proba[true_label_idx] for proba in estimator.predict_proba(df_data)]
    df_train = df[df['label'] != -1]

    # -- collectively extract best set of subject entities --
    df_train2 = df_train.iloc[train2_idxs]
    _train_selection_model(df_train2, config['sampling_functions'], config['selection'])
    return _extract_subject_entities(df, config['sampling_functions'], config['selection'])


def _train_selection_model(df: pd.DataFrame, sampling_funcs: list, selection_params: dict):
    """Train regression model for the collective evaluation of entity sets."""
    lps = _extract_listpage_data(df)
    lps = _sample_from_listings(lps, df, sampling_funcs)
    lp_data_samples = _run_multicore(_extract_training_data, [(lp, df.loc[lp.entities]) for lp in lps])
    data_samples = [s for samples in lp_data_samples for s in samples]
    df_sample = pd.DataFrame([s.stats for s in data_samples])
    selection_params['model'].fit(df_sample.drop(columns='label'), df_sample['label'])


def _extract_subject_entities(df: pd.DataFrame, sampling_funcs: list, selection_params: dict) -> dict:
    """Identify subject entities in list pages with collective extraction."""
    lps = _extract_listpage_data(df)
    lps = _sample_from_listings(lps, df, sampling_funcs)
    lps = _find_subject_entities(lps, df, selection_params)

    # Temporary storage of identified entities for development purposes
    # TODO: Remove after development of disambiguation & merging is done
    valid_entities = {idx for lp in lps for idx in lp.subject_entities}
    extraction_type = 'enum' if '_entry_idx' in df else 'table'
    df.loc[valid_entities, ['_listpage_uri', '_entity_uri', '_text', '_list_type']].to_csv(f'listpage_extracted-{extraction_type}-entities_v4.csv', sep=';', index=False)

    # TODO: Proper disambiguation and merging of identified entities

    return defaultdict(set, {lp.uri: set(df.loc[lp.subject_entities, '_entity_uri'].to_list()) for lp in lps})


# COLLECTIVE EXTRACTION
Line = namedtuple('Line', ['id', 'entities'])
EntitySample = namedtuple('EntitySample', ['types', 'entities', 'stats'])
Listing = namedtuple('Listing', ['id', 'lines', 'samples'])
Listpage = namedtuple('Listpage', ['uri', 'listings', 'entities', 'subject_entities'])


def _extract_listpage_data(df) -> list:
    """Convert list pages into structured data format."""
    lps = []
    for lp_uri, entity_indices in df.groupby('_listpage_uri').apply(lambda x: x.index.tolist()).to_dict().items():
        line_collections = defaultdict(lambda: defaultdict(set))
        for idx, row in df.loc[entity_indices].iterrows():
            line_collections[_get_listing_id(row)][row['_line_idx']].add(idx)
        listings = [Listing(id=cid, lines=[Line(*ldata) for ldata in cdata.items()], samples=[]) for cid, cdata in line_collections.items()]
        lps.append(Listpage(uri=lp_uri, listings=listings, entities=entity_indices, subject_entities=set()))
    return lps


def _get_listing_id(row) -> str:
    return row['_section_name'] if '_entry_idx' in row else f'{row["_section_name"]}_{row["_table_idx"]}'


# SAMPLING

def _sample_from_listings(lps: list, df: pd.DataFrame, sample_funcs: list) -> list:
    multicore_params = _sample_from_listings_param_generator(lps, df, sample_funcs)
    result = _run_multicore(_sample_from_listings_internal, multicore_params)
    return [lp for lp in result]


def _sample_from_listings_param_generator(lps: list, df: pd.DataFrame, sample_funcs: list):
    for lp in lps:
        yield lp, df.loc[lp.entities], sample_funcs


def _sample_from_listings_internal(lp: Listpage, df: pd.DataFrame, sample_funcs: list):
    # collect samples for listings
    for listing in lp.listings:
        for sample_func in sample_funcs:
            for sample_types, entities in sample_func(df, listing.lines):
                if entities:
                    listing.samples.append(EntitySample(sample_types, entities, {}))
    # remove listings without samples
    for l in [l for l in lp.listings if not l.samples]:
        lp.listings.remove(l)
    return lp


def _sample_by_entity_position(df: pd.DataFrame, lines: list) -> list:
    lines_by_depth = defaultdict(set)
    for _, row in df.iterrows():
        lines_by_depth[row['entry_depth']].add(row['_line_idx'])
    samples = []
    for dep, line_ids in lines_by_depth.items():
        dep_lines = [l for l in lines if l.id in line_ids]
        for ent_pos in range(2):
            samples.append(({f'ent-pos-dep_{ent_pos}_{dep}': 1}, [idx for line in dep_lines for idx in line.entities if df.at[idx, '_entity_line_idx'] == ent_pos]))
    return samples


def _sample_by_column(df: pd.DataFrame, lines: list) -> list:
    column_idxs = df['_column_idx'].unique()
    return [({f'col-first_{idx}': 1}, list(df.loc[(df['_column_idx'] == idx) & (df['_row_idx'] > 0) & (df['entity_first'])].index)) for idx in column_idxs]


# FEATURE GENERATION & EXTRACTION

def _extract_training_data(lp: Listpage, df: pd.DataFrame) -> list:
    # add all base samples to data
    lp_data_samples = [s for l in lp.listings for s in l.samples]
    # add some random combinations of samples as additional data points
    sample_sizes = [len(l.samples) for l in lp.listings]
    sample_count = int(min(np.sum(sample_sizes), np.product(sample_sizes)))
    for _ in range(sample_count):
        entities = set()
        for l in lp.listings:
            entities.update(random.choice(l.samples).entities)
        lp_data_samples.append(EntitySample({}, entities, {}))
    # compute features for samples
    for s in lp_data_samples:
        _compute_feature_proba(df, s)
        _compute_feature_tagsim(df, s)
        _compute_feature_similarents(df, s)
        _compute_feature_label(df, s)
    return lp_data_samples


def _compute_feature_label(df: pd.DataFrame, entity_sample: EntitySample):
    entity_sample.stats['label'] = df.loc[entity_sample.entities, 'label'].mean()


def _find_subject_entities(lps: list, df: pd.DataFrame, selection_params: dict) -> list:
    multicore_params = _find_subject_entities_param_generator(lps, df, selection_params)
    result = _run_multicore(_find_subject_entities_internal, multicore_params)
    return [lp for lp in result]


def _find_subject_entities_param_generator(lps: list, df: pd.DataFrame, selection_params: dict):
    for lp in lps:
        yield lp, df.loc[lp.entities], selection_params


def _find_subject_entities_internal(lp: Listpage, df: pd.DataFrame, selection_params: dict):
    current_samples = []
    for listing in lp.listings:
        # select best samples from listing
        listing_samples = [(s, _compute_sample_score(df, s, selection_params['model'])) for s in listing.samples]
        listing_samples = _select_top_samples(listing_samples, selection_params['n_candidates'], selection_params['min_score'])
        # produce combined samples
        if not listing_samples:
            continue
        if not current_samples:
            current_samples = listing_samples
            continue
        combined_samples = _generate_combined_samples(current_samples, listing_samples)
        combined_samples = [(cs, _compute_sample_score(df, cs, selection_params['model'])) for cs in combined_samples]
        # take best combined samples into next round
        current_samples = _select_top_samples(combined_samples, selection_params['n_candidates'], selection_params['min_score'])
    entity_set = max(current_samples, key=lambda x: x[1])[0].entities if current_samples else set()
    lp.subject_entities.update(entity_set)
    return lp


def _generate_combined_samples(current_samples, new_samples) -> list:
    combined_samples = []
    for cs, _ in current_samples:
        for ns, _ in new_samples:
            types = defaultdict(int, cs.types)
            for t, c in ns.types.items():
                types[t] += c
            combined_samples.append(EntitySample(types, cs.entities + ns.entities, {}))
    return combined_samples


def _compute_sample_score(df: pd.DataFrame, entity_sample: EntitySample, selection_model) -> float:
    # compute relevant stats
    _compute_feature_proba(df, entity_sample)
    _compute_feature_tagsim(df, entity_sample)
    _compute_feature_similarents(df, entity_sample)
    # compute score
    return selection_model.predict(pd.DataFrame([entity_sample.stats]))[0]


def _compute_feature_proba(df: pd.DataFrame, entity_sample: EntitySample):
    entity_sample.stats['proba'] = df.loc[entity_sample.entities, 'proba'].mean()


def _compute_feature_tagsim(df: pd.DataFrame, entity_sample: EntitySample):
    tag_columns = [c for c in df.columns if c.startswith('entity_ne')]
    tag_counts = df.loc[entity_sample.entities, tag_columns].sum().values
    entity_sample.stats['tag_sim'] = math.sqrt(((tag_counts / tag_counts.sum()) ** 2).sum()) if tag_counts.sum() > 0 else 0


def _compute_feature_similarents(df: pd.DataFrame, entity_sample: EntitySample):
    all_entities = []
    for _, row in df[df.index.isin(entity_sample.entities)].iterrows():
        entity_uri = row['_entity_uri']
        if row['_link_type'] == 'grey':
            all_entities.append(entity_uri[entity_uri.rfind('__')+2:])
        else:
            all_entities.append(entity_uri[entity_uri.rfind('/')+1:])
    entity_sample.stats['similar_ents'] = len(set(all_entities)) / len(all_entities)


def _select_top_samples(samples: list, n_candidates: int, min_score: float) -> list:
    samples = [s for s in samples if s[1] > min_score]
    return sorted(samples, key=lambda x: x[1], reverse=True)[:min(len(samples), n_candidates)]


# MISC

def _run_multicore(func, params):
    pool = mp.Pool(util.get_config('max_cpus'))
    results = pool.starmap(func, params)
    pool.close()
    return results
