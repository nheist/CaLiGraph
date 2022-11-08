from typing import List, Union, Set
from collections import defaultdict
from tqdm import tqdm
import torch
import queue
import hnswlib
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from sentence_transformers.util import dot_score
import utils
from impl.util.transformer import SpecialToken
from impl.wikipedia.page_parser import MentionId
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from entity_linking.entity_disambiguation.data import DataCorpus, Pair, CandidateAlignment


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


def generate_training_data(scenario: MatchingScenario, data_corpus: DataCorpus, negatives: List[Pair], add_page_context: bool, add_text_context: bool, add_entity_abstract: bool, add_kg_info: bool) -> List[InputExample]:
    mention_input, mention_known = data_corpus.get_mention_input(add_page_context, add_text_context)
    if scenario == MatchingScenario.MENTION_MENTION:
        mention_input = {m_id: m_input for m_id, m_input in mention_input.items() if mention_known[m_id]}
        target_input = mention_input
    else:  # scenario: MENTION_ENTITY
        target_input = data_corpus.get_entity_input(add_entity_abstract, add_kg_info)
    positives = data_corpus.alignment.sample_matches(scenario)
    input_examples = [InputExample(texts=[mention_input[mention_id], target_input[target_id]], label=1) for mention_id, target_id in positives]
    input_examples.extend([InputExample(texts=[mention_input[mention_id], target_input[target_id]], label=0) for mention_id, target_id in negatives])
    return input_examples


# NEAREST-NEIGHBOR SEARCH


# optimized version of function in sentence_transformers.util
def paraphrase_mining_embeddings(ca: CandidateAlignment, embeddings: torch.Tensor, mention_ids: List[MentionId], query_chunk_size: int = 5000, corpus_chunk_size: int = 100000, max_pairs: int = 500000, top_k: int = 100, add_best: bool = False):
    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs
    best_pairs_per_item = defaultdict(lambda: (None, 0.0))
    top_pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in tqdm(range(0, len(embeddings), corpus_chunk_size), total=len(embeddings) // corpus_chunk_size, desc='MM nn-search'):
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            scores = dot_score(embeddings[query_start_idx:query_start_idx+query_chunk_size], embeddings[corpus_start_idx:corpus_start_idx+corpus_chunk_size])

            scores_top_k_values, scores_top_k_idx = torch.topk(scores, min(top_k, len(scores[0])), dim=1, largest=True, sorted=False)
            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(scores)):
                for top_k_idx, corpus_itr in enumerate(scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr
                    if i == j:
                        continue
                    score = scores_top_k_values[query_itr][top_k_idx]
                    if add_best and score > .5:
                        # collect best pairs per item
                        if best_pairs_per_item[i][1] < score:
                            best_pairs_per_item[i] = (j, score)
                        if best_pairs_per_item[j][1] < score:
                            best_pairs_per_item[j] = (i, score)
                    if score > min_score:
                        # collect overall top pairs
                        top_pairs.put((score, i, j))
                        num_added += 1
                        if num_added >= max_pairs:
                            entry = top_pairs.get()
                            min_score = entry[0]
    # assemble the final pairs
    if add_best:
        for i, (j, score) in best_pairs_per_item.items():
            ca.add_candidate((mention_ids[i], mention_ids[j]), score)
    while not top_pairs.empty():
        score, i, j = top_pairs.get()
        ca.add_candidate((mention_ids[i], mention_ids[j]), score)


# optimized version of function in sentence_transformers.util
def semantic_search(ca: CandidateAlignment, query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor, mention_ids: List[MentionId], entity_ids: List[int], query_chunk_size: int = 100, corpus_chunk_size: int = 500000, top_k: int = 10):
    # check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)
    # collect documents per query
    queries_result_list = defaultdict(dict)
    for query_start_idx in tqdm(range(0, len(query_embeddings), query_chunk_size), total=len(query_embeddings) // query_chunk_size, desc='ME nn-search'):
        # iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # compute similarities
            cos_scores = dot_score(query_embeddings[query_start_idx:query_start_idx+query_chunk_size], corpus_embeddings[corpus_start_idx:corpus_start_idx+corpus_chunk_size])
            # get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id][corpus_id] = score
    # sort and strip to top_k results; convert to original indices
    for mention_idx, entity_scores in queries_result_list.items():
        top_k_entity_scores = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for ent_idx, score in top_k_entity_scores:
            ca.add_candidate((mention_ids[mention_idx], entity_ids[ent_idx]), score)


def approximate_paraphrase_mining_embeddings(ca: CandidateAlignment, mention_embeddings: torch.Tensor, mention_ids: List[MentionId], max_pairs: int = 500000, top_k: int = 100, add_best: bool = False):
    index = _build_ann_index(mention_embeddings, 400, 64, 100)

    top_k += 1  # a sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs
    best_pairs_per_item = defaultdict(lambda: (None, 0.0))
    top_pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0
    for idx_a, mention_emb in tqdm(enumerate(mention_embeddings), desc='MM ann-search'):
        mention_a = mention_ids[idx_a]
        for idx_b, dist in zip(*index.knn_query(mention_emb, k=top_k)):
            mention_b = mention_ids[idx_b]
            if mention_a == mention_b:
                continue
            score = 1 - dist
            if add_best and score > .5:
                # collect best pairs per item
                if best_pairs_per_item[mention_a][1] < score:
                    best_pairs_per_item[mention_a] = (mention_b, score)
                if best_pairs_per_item[mention_b][1] < score:
                    best_pairs_per_item[mention_b] = (mention_a, score)
            if score > min_score:
                # collect overall top pairs
                top_pairs.put((score, *sorted([mention_a, mention_b])))
                num_added += 1
                if num_added >= max_pairs:
                    entry = top_pairs.get()
                    min_score = entry[0]
    # assemble the final pairs
    if add_best:
        for mention_a, (mention_b, score) in best_pairs_per_item.items():
            ca.add_candidate((mention_a, mention_b), score)
    while not top_pairs.empty():
        score, mention_a, mention_b = top_pairs.get()
        ca.add_candidate((mention_a, mention_b), score)


def approximate_semantic_search(ca: CandidateAlignment, mention_embeddings: torch.Tensor, entity_embeddings: torch.Tensor, mention_ids: List[MentionId], entity_ids: List[int], top_k: int = 10):
    index = _build_ann_index(entity_embeddings, 400, 64, 50)
    for mention_idx, mention_emb in tqdm(enumerate(mention_embeddings), desc='ME ann-search'):
        m_id = mention_ids[mention_idx]
        for ent_idx, dist in zip(*index.knn_query(mention_emb, k=top_k)):
            e_id = entity_ids[ent_idx]
            ca.add_candidate((m_id, e_id), 1 - dist)


def _build_ann_index(embeddings: torch.Tensor, ef_construction: int, M: int, ef: int) -> hnswlib.Index:
    utils.get_logger().debug('Building ANN index..')
    index = hnswlib.Index(space='ip', dim=embeddings.shape[-1])
    index.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=M)
    index.add_items(embeddings, list(range(len(embeddings))))
    index.set_ef(ef)
    return index
