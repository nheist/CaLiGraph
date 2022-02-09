import re
import unidecode
from nltk.corpus import stopwords
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import entity_linking.util as el_util


def get_sf_to_entity_mapping(entity_occurrence_data: pd.DataFrame):
    page_sfs = set(entity_occurrence_data['_text'].unique())  # surface forms from pages
    # entities and surface forms from graph
    graph_sf_to_entity_mapping = defaultdict(set)
    for ent_uri, sfs in dbp_store.get_all_lexicalisations().items():
        ent = dbp_util.resource2name(ent_uri)
        for sf in sfs:
            graph_sf_to_entity_mapping[sf].add(ent)
    return _block_entities_by_surface_forms(page_sfs, graph_sf_to_entity_mapping, n_cores=el_util.MAX_CORES)


def _block_entities_by_surface_forms(page_sfs: set, graph_sf_to_entity_mapping: dict, n_cores=1) -> dict:
    nsf_to_entities_mapping = defaultdict(set)

    with mp.Pool(processes=n_cores) as pool:
        # collect all normalised surface forms that are considered for blocking
        page_nsfs = {nsf for nsf in tqdm(pool.imap_unordered(_normalise_sf, page_sfs, chunksize=10000), desc='Normalizing page sfs', total=len(page_sfs))}
        # find entities in the graph that match the nsfs of pages
        for nsf, entities in tqdm(pool.imap_unordered(_normalise_sf_for_ents, graph_sf_to_entity_mapping.items(), chunksize=10000), desc='Processing graph entities', total=len(graph_sf_to_entity_mapping)):
            if nsf not in page_nsfs:
                continue
            nsf_to_entities_mapping[nsf].update(entities)
    return nsf_to_entities_mapping


def _normalise_sf_for_ents(sf_with_entities: tuple) -> tuple:
    sf, entities = sf_with_entities
    return _normalise_sf(sf), entities


def _normalise_sf(sf: str) -> tuple:
    sf = unidecode.unidecode(sf.lower())  # lowercase everything and convert special characters
    word_parts = {_make_alphanum(w) for w in sf.split() if w}  # remove non-alphanumeric characters
    filtered_word_parts = word_parts.difference(stopwords.words('english'))
    return tuple(sorted(filtered_word_parts or word_parts))


def _make_alphanum(text: str) -> str:
    text_alphanum = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text_alphanum if len(text_alphanum) > 2 else text
