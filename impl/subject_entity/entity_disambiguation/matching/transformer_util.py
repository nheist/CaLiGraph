from typing import List, Union
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import hnswlib
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from sentence_transformers.util import dot_score
import utils
from impl.util.transformer import SpecialToken
from impl.wikipedia.page_parser import MentionId
from impl.subject_entity.entity_disambiguation.matching.util import MatchingScenario
from impl.subject_entity.entity_disambiguation.data import DataCorpus, Pair, CandidateAlignment


def add_special_tokens(model: Union[SentenceTransformer, CrossEncoder]):
    if isinstance(model, SentenceTransformer):
        word_embedding_model = model._first_module()
        tokenizer = word_embedding_model.tokenizer
        transformer = word_embedding_model.auto_model
    elif isinstance(model, CrossEncoder):
        tokenizer = model.tokenizer
        transformer = model.model
    else:
        raise ValueError(f'Invalid type for model: {type(model)}')
    tokenizer.add_tokens(list(SpecialToken.all_tokens()), special_tokens=True)
    transformer.resize_token_embeddings(len(tokenizer))


def generate_training_data(scenario: MatchingScenario, data_corpus: DataCorpus, sample_size: int, negatives: List[Pair], add_page_context: bool, add_text_context: bool, add_entity_abstract: bool, add_kg_info: bool) -> List[InputExample]:
    utils.get_logger().debug(f'Generating training data for scenario {scenario.name}..')
    mention_input, mention_known = data_corpus.get_mention_input(add_page_context, add_text_context)
    if scenario == MatchingScenario.MENTION_MENTION:
        mention_input = {m_id: m_input for m_id, m_input in mention_input.items() if mention_known[m_id]}
        target_input = mention_input
        positives = data_corpus.alignment.sample_mm_matches(sample_size)
    else:  # scenario: MENTION_ENTITY
        target_input = data_corpus.get_entity_input(add_entity_abstract, add_kg_info)
        positives = data_corpus.alignment.sample_me_matches(sample_size)
    input_examples = [InputExample(texts=[mention_input[mention_id], target_input[target_id]], label=1) for mention_id, target_id in positives]
    input_examples.extend([InputExample(texts=[mention_input[mention_id], target_input[target_id]], label=0) for mention_id, target_id in negatives])
    return input_examples


# NEAREST-NEIGHBOR SEARCH


# optimized version of function in sentence_transformers.util
def semantic_search(ca: CandidateAlignment, query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor, query_ids: List[MentionId], corpus_ids: List[Union[MentionId, int]], query_chunk_size: int = 100, corpus_chunk_size: int = 500000, top_k: int = 10):
    # check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)
    # collect documents per query
    queries_result_list = defaultdict(dict)
    for query_start_idx in tqdm(range(0, len(query_embeddings), query_chunk_size), total=len(query_embeddings) // query_chunk_size, desc='semantic search'):
        # iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # compute similarities
            cos_scores = dot_score(query_embeddings[query_start_idx:query_start_idx+query_chunk_size], corpus_embeddings[corpus_start_idx:corpus_start_idx+corpus_chunk_size])
            # get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            for query_itr in range(len(cos_scores)):
                query_id = query_ids[query_start_idx + query_itr]
                for sub_corpus_idx, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_idx]
                    queries_result_list[query_id][corpus_id] = score
    # sort and strip to top_k results; convert to original indices
    for query_id, corpus_scores in queries_result_list.items():
        top_k_entity_scores = sorted(corpus_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for corpus_id, score in top_k_entity_scores:
            ca.add_candidate((query_id, corpus_id), score)


def approximate_semantic_search(ca: CandidateAlignment, query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, query_ids: List[MentionId], corpus_ids: List[Union[MentionId, int]], top_k: int = 10):
    index = _build_ann_index(corpus_embeddings, 400, 64, 50)
    utils.get_logger().debug('Running approximate semantic search..')
    for query_idx, (corpus_indices, distances) in enumerate(zip(*index.knn_query(query_embeddings, k=top_k))):
        for ent_idx, dist in zip(corpus_indices, distances):
            ca.add_candidate((query_ids[query_idx], corpus_ids[ent_idx]), 1 - dist)


def _build_ann_index(embeddings: np.ndarray, ef_construction: int, M: int, ef: int) -> hnswlib.Index:
    utils.get_logger().debug('Building ANN index..')
    index = hnswlib.Index(space='ip', dim=embeddings.shape[-1])
    index.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=M)
    index.add_items(embeddings, list(range(len(embeddings))))
    index.set_ef(ef)
    return index
