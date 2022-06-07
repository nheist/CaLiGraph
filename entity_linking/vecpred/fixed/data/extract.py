from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
import entity_linking.vp_utils as el_util
from typing import Tuple
import impl.dbpedia.util as dbp_util
from impl import wikipedia
from impl.subject_entity import extract, combine
from impl.subject_entity.preprocess.word_tokenize import WordTokenizer
from entity_linking.vecpred.fixed.data import blocking, load
import operator


def extract_training_data(parts: int, embedding_type: str, extract_validation_set: bool):
    df_listings, dataset_pages_chunks = _get_listings_and_page_chunks()

    if extract_validation_set:
        _extract_validation_set(dataset_pages_chunks)

    # extract / load entity occurrence data
    entity_occurrence_data = _get_entity_occurrence_data(parts, df_listings, dataset_pages_chunks)

    # block surface forms
    sf_to_entity_mapping = blocking.get_sf_to_entity_mapping(entity_occurrence_data)  # TODO: here we return entity indices -> adapt remaining program to that
    el_util.store_data(sf_to_entity_mapping, 'sf-to-entity-word-mapping.p', parts=parts)

    # load embedding vectors
    entity_vecs = el_util.load_entity_vectors(embedding_type)

    # create X
    X = _create_X(entity_occurrence_data, entity_vecs)
    el_util.store_data(X, 'X.feather', parts=parts, embedding_type=embedding_type)

    # filter entity vectors to keep only relevant ones
    relevant_entities = set(X['_ent']) | {e for ents in sf_to_entity_mapping.values() for e in ents}
    entity_vecs = entity_vecs[entity_vecs['ent'].isin(relevant_entities)].copy().reset_index(drop=True)
    el_util.store_data(entity_vecs, 'entity-vectors.feather', parts=parts, embedding_type=embedding_type)

    # retrieve entity vector indices
    idx2ent, ent2idx = load._get_entity_vector_indices(entity_vecs)

    # create Y
    Y = _create_Y(X, ent2idx, sf_to_entity_mapping)
    el_util.store_data(Y, 'Y.p', parts=parts, embedding_type=embedding_type)


def _get_listings_and_page_chunks() -> Tuple[pd.DataFrame, np.ndarray]:
    df_listings = pd.read_csv(f'{el_util.DATA_PATH}/clgv21_distant-supervision-listings_full.csv', sep=';')
    all_pages = df_listings['p'].unique()
    np.random.shuffle(all_pages)
    return df_listings, np.array_split(all_pages, el_util.DATASET_PARTS)


def _extract_validation_set(dataset_pages_chunks: np.ndarray):
    val_pages = dataset_pages_chunks[0][:int(len(dataset_pages_chunks[0]) * el_util.DATASET_VALIDATION_SIZE)]
    el_util.store_data(val_pages, 'validation-set-pages.p')


def _get_entity_occurrence_data(parts: int, df_listings: pd.DataFrame, dataset_pages_chunks: np.ndarray):
    for i in range(parts):
        # extract entity occurrences if not already existing
        filename = f'entity-occurrences-{i}.feather'
        if not el_util.file_exists(filename):
            _extract_entity_occurrences(df_listings, dataset_pages_chunks[i], filename)

    return pd.concat([el_util.load_data(f'entity-occurrences-{i}.feather') for i in range(parts)], ignore_index=True)


def _extract_entity_occurrences(df_listings: pd.DataFrame, dataset_pages: np.ndarray, filename: str):
    valid_pages = set(dataset_pages)
    valid_listings = _get_dataset_listings(df_listings, dataset_pages)
    tokenizer, model = extract.get_tagging_tokenizer_and_model(lambda: None)

    wikipedia_pages = {p: m for p, m in wikipedia.get_wikipedia_pages().items() if dbp_util.resource_iri2name(p) in valid_pages}

    result_data_columns = ['_ent', '_text', '_page', '_section', 'e_tag'] + [f'e_{i}' for i in range(768)] + [f'l_{i}' for i in range(768)]
    result_data = []

    word_tokenizer = WordTokenizer()
    for page_uri, page_markup in tqdm(wikipedia_pages.items(), desc=f'Extracting {filename}'):
        page_name = dbp_util.resource_iri2name(page_uri)
        # TODO: as soon as WS issue (old cache) is resolved, we can retrieve page_batches from subject_entity._get_page_data()
        page_chunks = word_tokenizer({page_uri: page_markup})[page_uri]

        subject_entities = extract.extract_subject_entities(page_chunks, tokenizer, model)
        enriched_entities = combine._match_entities_for_page(page_uri, subject_entities, page_markup)

        for ts, ts_data in enriched_entities.items():
            for s, s_data in ts_data.items():
                if (page_name, ts, s) not in valid_listings:
                    continue  # skip listings that are not in distant-supervision data
                listing_vector = list(subject_entities[1][ts][s]['_embedding'])
                for ent_name, ent_data in s_data.items():
                    ent_text = ent_data['text']
                    ent_tag = ent_data['tag']
                    entity_vector = list(subject_entities[1][ts][s][ent_text])
                    result_data.append([ent_name, ent_text, page_name, s, ent_tag] + entity_vector + listing_vector)
    df = pd.DataFrame(columns=result_data_columns, data=result_data)
    el_util.store_data(df, filename)


def _get_dataset_listings(df_listings: pd.DataFrame, dataset_pages: np.ndarray) -> set:
    dataset_listings = df_listings[df_listings['p'].isin(dataset_pages)]
    return set(map(tuple, dataset_listings.to_numpy()))


def _create_X(entity_occurrence_data: pd.DataFrame, entity_vecs: pd.DataFrame) -> pd.DataFrame:
    # add page vectors
    X = pd.merge(left=entity_occurrence_data, left_on='_page', right=entity_vecs, right_on='ent').drop(columns='ent')
    X.rename(inplace=True, columns={f'v_{i}': f'p_{i}' for i in range(200)})
    # 1-hot encode entity tags
    tag_dummies = pd.get_dummies(X['e_tag'], prefix='tag')
    X = pd.merge(X.drop(columns='e_tag'), tag_dummies, left_index=True, right_index=True)
    return X


def _create_Y(X, ent2idx, sf_to_entity_mapping, padding_val=-1, entity_count=100) -> np.ndarray:
    with mp.Pool(processes=el_util.MAX_CORES) as pool:
        entity_occurrences = {t for t in tqdm(pool.imap_unordered(_normalise_row_text, X[['_ent', '_text']].itertuples(name=None), chunksize=10000), total=len(X), desc='Normalising surface forms')}
    Y = []
    for _, ent, nsf in sorted(tqdm(entity_occurrences, desc='Collecting indices'), key=operator.itemgetter(0)):
        pe_index = ent2idx[ent] if ent in ent2idx else padding_val
        # compute indices of negative entities
        negative_entities = sf_to_entity_mapping[nsf].difference({ent})
        ne_indices = [ent2idx[e] for e in negative_entities if e in ent2idx]  # filter invalid ents and convert to idx
        entity_indices = [pe_index] + ne_indices[:min(len(ne_indices), entity_count-1)]
        padded_entity_indices = np.pad(entity_indices, (0, entity_count-len(entity_indices)), constant_values=padding_val)
        Y.append(padded_entity_indices)
    return np.array(Y)


def _normalise_row_text(row: tuple) -> tuple:
    idx, ent, text = row
    return idx, ent, blocking._normalise_sf(text)
