"""Extraction of type lexicalisations from the Wikipedia corpus via NIF files."""

import utils
from typing import Tuple
import pynif
import bz2
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from spacy.lang.en.stop_words import STOP_WORDS
import impl.dbpedia.store as dbp_store
import impl.util.nlp as nlp_util
import impl.util.spacy as spacy_util


def extract_wiki_corpus_resources():
    """Crawl the Wikipedia corpus for hearst patterns to retrieve hypernyms and type lexicalisations."""
    if utils.load_cache('wikipedia_type_lexicalisations') is not None:
        return  # only compute hypernyms and type lexicalisations if they are not existing already

    utils.get_logger().info('WIKIPEDIA/NIF: Computing wikipedia hypernyms and type lexicalisations..')
    total_hypernyms = defaultdict(lambda: defaultdict(int))
    total_type_lexicalisations = defaultdict(lambda: defaultdict(int))

    # initialize some caches to reduce the setup time of the individual processes
    dbp_store.get_types('')
    dbp_store.get_inverse_lexicalisations('')
    spacy_util.get_hearst_pairs('')

    with mp.Pool(processes=utils.get_config('max_cpus')) as pool:
        for hypernyms, type_lexicalisations in pool.imap_unordered(_compute_counts_for_resource, tqdm(_retrieve_plaintexts()), chunksize=1000):
            for (sub, obj), count in hypernyms.items():
                total_hypernyms[sub][obj] += count
            for (sub, obj), count in type_lexicalisations.items():
                total_type_lexicalisations[sub][obj] += count

    wikipedia_hypernyms = {word: dict(hypernym_counts) for word, hypernym_counts in total_hypernyms.items()}
    utils.update_cache('wikipedia_hypernyms', wikipedia_hypernyms)

    type_lexicalisations = {word: dict(type_counts) for word, type_counts in total_type_lexicalisations.items() if word not in STOP_WORDS}
    utils.update_cache('wikipedia_type_lexicalisations', type_lexicalisations)


def _compute_counts_for_resource(uri_with_text: tuple) -> tuple:
    uri, text = uri_with_text
    hypernyms = defaultdict(int)
    type_lexicalisations = defaultdict(int)
    for sub, obj in spacy_util.get_hearst_pairs(text):
        # collect hypernym statistics in Wikipedia
        hypernyms[(nlp_util.lemmatize_token(sub.root).lower(), nlp_util.lemmatize_token(obj.root).lower())] += 1

        # for each word, count the types that it refers to
        if uri not in dbp_store.get_inverse_lexicalisations(sub.text):
            continue  # discard, if the resource text does not refer to the subject of the article
        for t in dbp_store.get_types(uri):
            for word in obj:
                type_lexicalisations[(nlp_util.lemmatize_token(word).lower(), t)] += 1
    return hypernyms, type_lexicalisations


def _retrieve_plaintexts() -> Tuple[str, str]:
    """Return an iterator over DBpedia resources and their Wikipedia plaintexts."""
    with bz2.open(utils.get_data_file('files.dbpedia.nif_context'), mode='rb') as nif_file:
        nif_collection = pynif.NIFCollection.loads(nif_file.read(), format='turtle')
        for nif_context in nif_collection.contexts:
            resource_uri = nif_context.original_uri[:nif_context.original_uri.rfind('?')]
            # remove parentheses and line breaks from text for easier parsing
            resource_plaintext = nif_context.mention.replace('\n', ' ')
            resource_plaintext = nlp_util.remove_bracket_content(resource_plaintext, substitute='')
            resource_plaintext = nlp_util.remove_bracket_content(resource_plaintext, bracket_type='[', substitute='')
            yield resource_uri, resource_plaintext
