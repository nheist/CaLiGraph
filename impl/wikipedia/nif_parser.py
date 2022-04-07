"""Extraction of type lexicalisations from the Wikipedia corpus via NIF files."""

from typing import Tuple
from collections import defaultdict, Counter
import utils
from utils import get_logger
import pynif
import bz2
import multiprocessing as mp
from tqdm import tqdm
from spacy.lang.en.stop_words import STOP_WORDS
import impl.util.rdf as rdf_util
import impl.util.nlp as nlp_util
import impl.util.spacy as spacy_util
from impl.dbpedia.resource import DbpEntity, DbpResourceStore


def extract_wiki_corpus_resources():
    """Crawl the Wikipedia corpus for hearst patterns to retrieve hypernyms and type lexicalisations."""
    if utils.load_cache('wikipedia_type_lexicalisations') is not None:
        get_logger().info('Skipping computation of hypernyms and type lexicalisations.')
        return  # only compute hypernyms and type lexicalisations if they are not existing already

    get_logger().info('Computing wikipedia hypernyms and type lexicalisations..')
    total_hypernyms = defaultdict(Counter)
    total_type_lexicalisations = defaultdict(Counter)

    # initialize some caches to reduce the setup time of the individual processes
    dbr = DbpResourceStore.instance()
    dbr.get_types(dbr.get_resource_by_idx(0))
    dbr.get_surface_form_references('')
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


def _compute_counts_for_resource(entity_with_text: Tuple[DbpEntity, str]) -> Tuple[dict, dict]:
    dbr = DbpResourceStore.instance()
    ent, text = entity_with_text
    hypernyms = Counter()
    type_lexicalisations = Counter()
    for sub, obj in spacy_util.get_hearst_pairs(text):
        # collect hypernym statistics in Wikipedia
        hypernyms[(nlp_util.lemmatize_token(sub.root).lower(), nlp_util.lemmatize_token(obj.root).lower())] += 1

        # for each word, count the types that it refers to
        if sub.text.lower() not in {'he', 'she', 'it'} or ent not in dbr.get_surface_form_references(sub.text):
            continue  # discard, if the resource text does not refer to the subject of the article
        for t in ent.get_types():
            for word in obj:
                type_lexicalisations[(nlp_util.lemmatize_token(word).lower(), t)] += 1
    return hypernyms, type_lexicalisations


def _retrieve_plaintexts():
    """Return an iterator over DBpedia resources and their Wikipedia plaintexts."""
    dbr = DbpResourceStore.instance()
    with bz2.open(utils.get_data_file('files.dbpedia.nif_context'), mode='rb') as nif_file:
        nif_collection = pynif.NIFCollection.loads(nif_file.read(), format='turtle')
    for nif_context in nif_collection.contexts:
        resource_iri = rdf_util.uri2iri(nif_context.original_uri[:nif_context.original_uri.rfind('?')])
        if not dbr.has_resource_with_iri(resource_iri):
            continue
        res = dbr.get_resource_by_iri(resource_iri)
        if not isinstance(res, DbpEntity) or res.is_meta:
            continue
        # remove parentheses and line breaks from text for easier parsing
        resource_plaintext = nif_context.mention.replace('\n', ' ')
        resource_plaintext = nlp_util.remove_bracket_content(resource_plaintext, substitute='')
        resource_plaintext = nlp_util.remove_bracket_content(resource_plaintext, bracket_type='[', substitute='')
        yield res, resource_plaintext
