from typing import List, Tuple, Dict, Union, Set
from collections import defaultdict
from itertools import cycle, islice
import torch
import queue
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from sentence_transformers.util import dot_score
import utils
from impl.util.string import alternate_iters_to_string
from impl.util.transformer import SpecialToken, EntityIndex
from impl.wikipedia.page_parser import WikiListing, WikiTable, WikiListingItem, WikiEnumEntry, MentionId
from impl.dbpedia.resource import DbpResourceStore
from impl.dbpedia.category import DbpCategoryStore
from impl.caligraph.entity import ClgEntity
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from entity_linking.entity_disambiguation.data import DataCorpus, Pair


CXS = SpecialToken.CONTEXT_SEP.value
CXE = SpecialToken.CONTEXT_END.value
COL = SpecialToken.TABLE_COL.value
ROW = SpecialToken.TABLE_ROW.value


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


def generate_training_data(scenario: MatchingScenario, data_corpus: DataCorpus, negatives: List[Pair], add_page_context: bool, add_category_context: bool, add_listing_entities: bool, add_entity_abstract: bool, add_kg_info: bool) -> List[InputExample]:
    source_input, source_known = prepare_listing_items(data_corpus.get_listings(), add_page_context, add_category_context, add_listing_entities)
    if scenario == MatchingScenario.MENTION_MENTION:
        source_input = {m_id: m_input for m_id, m_input in source_input.items() if source_known[m_id]}
        target_input = source_input
        positives = data_corpus.mm_alignment
    else:  # scenario: MENTION_ENTITY
        target_input = prepare_entities(data_corpus.get_entities(), add_entity_abstract, add_kg_info)
        positives = data_corpus.me_alignment
    input_examples = [InputExample(texts=[source_input[source_id], target_input[target_id]], label=1) for source_id, target_id, _ in positives]
    input_examples.extend([InputExample(texts=[source_input[source_id], target_input[target_id]], label=0) for source_id, target_id, _ in negatives])
    return input_examples


def prepare_listing_items(listings: List[WikiListing], add_page_context: bool, add_category_context: bool, add_listing_entities: bool) -> Tuple[Dict[MentionId, str], Dict[MentionId, bool]]:
    utils.get_logger().debug('Preparing listing items..')
    result = {}
    result_ent_known = {}
    if not add_page_context and not add_listing_entities:
        for l in listings:
            for i in l.get_items(has_subject_entity=True):
                mention_id = MentionId(l.page_idx, l.idx, i.idx)
                se = i.subject_entity
                result[mention_id] = f'{se.label} {SpecialToken.get_type_token(se.entity_type)}'
                result_ent_known[mention_id] = se.entity_idx != EntityIndex.NEW_ENTITY.value
        return result, result_ent_known
    for listing in listings:
        prepared_context = _prepare_listing_context(listing, add_category_context)
        prepared_items = [_prepare_listing_item(item) for item in listing.get_items()]
        for idx, item in enumerate(listing.get_items(has_subject_entity=True)):
            mention_id = MentionId(listing.page_idx, listing.idx, item.idx)
            item_se = item.subject_entity
            # add subject entity, its type, and page context
            item_content = f' {CXS} '.join([f'{item_se.label} {SpecialToken.get_type_token(item_se.entity_type)}', prepared_context])
            # add item and `add_listing_entities` subsequent items (add items from start if no subsequent items left)
            item_content += ''.join(islice(cycle(prepared_items), idx, idx + add_listing_entities + 1))
            result[mention_id] = item_content
            result_ent_known[mention_id] = item_se.entity_idx != EntityIndex.NEW_ENTITY.value
    return result, result_ent_known


def _prepare_listing_context(listing: WikiListing, add_category_context: bool) -> str:
    res = DbpResourceStore.instance().get_resource_by_idx(listing.page_idx)
    res_description = f'{res.get_label()} {SpecialToken.get_type_token(res.get_type_label())}'
    if add_category_context:
        cats = list(DbpCategoryStore.instance().get_categories_for_resource(res.idx))[:3]
        if cats:
            res_description += ' ' + ', '.join([cat.get_label() for cat in cats])
    ctx = [res_description, listing.topsection.title, listing.section.title]
    if isinstance(listing, WikiTable):
        ctx.append(_prepare_listing_item(listing.header))
    return f' {CXS} '.join(ctx) + f' {CXE} '


def _prepare_listing_item(item: WikiListingItem) -> str:
    if isinstance(item, WikiEnumEntry):
        tokens = [SpecialToken.get_entry_by_depth(item.depth)] + item.tokens
        whitespaces = [' '] + item.whitespaces[:-1] + [' ']
    else:  # WikiTableRow
        tokens, whitespaces = [], []
        for cell_tokens, cell_whitespaces in zip(item.tokens, item.whitespaces):
            tokens += [COL] + cell_tokens
            whitespaces += [' '] + cell_whitespaces[:-1] + [' ']
        tokens[0] = ROW  # special indicator for start of table row
    return alternate_iters_to_string(tokens, whitespaces)


def prepare_entities(entities: Set[ClgEntity], add_entity_abstract: bool, add_kg_info: int) -> Dict[int, str]:
    utils.get_logger().debug('Preparing entities..')
    result = {}
    for e in entities:
        ent_description = [f'{e.get_label()} {SpecialToken.get_type_token(e.get_type_label())}']
        if add_entity_abstract:
            ent_description.append((e.get_abstract() or '')[:200])
        if add_kg_info:
            kg_info = [f'type = {t.get_label()}' for t in e.get_types()]
            prop_count = max(0, add_kg_info - len(kg_info))
            if prop_count > 0:
                props = list(e.get_properties(as_tuple=True))[:prop_count]
                kg_info += [f'{pred.get_label()} = {val.get_label() if isinstance(val, ClgEntity) else val}' for pred, val in props]
        result[e.idx] = f' {CXS} '.join(ent_description)
    return result


# optimized version of function in sentence_transformers.util
def paraphrase_mining_embeddings(embeddings: torch.Tensor, mention_ids: List[MentionId], query_chunk_size: int = 5000, corpus_chunk_size: int = 100000, max_pairs: int = 500000, top_k: int = 100, add_best: bool = False) -> Set[Pair]:
    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs
    best_pairs_per_item = defaultdict(lambda: (None, 0.0))
    top_pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
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

    # Assemble the final pairs
    pairs = {Pair(*sorted([mention_ids[i], mention_ids[j]]), score) for i, (j, score) in best_pairs_per_item.items()} if add_best else set()
    while not top_pairs.empty():
        score, i, j = top_pairs.get()
        pairs.add(Pair(*sorted([mention_ids[i], mention_ids[j]]), score))
    return pairs


# optimized version of function in sentence_transformers.util
def semantic_search(query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor, mention_ids: List[MentionId], entity_ids: List[int], query_chunk_size: int = 100, corpus_chunk_size: int = 500000, top_k: int = 10) -> Set[Pair]:
    # check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)
    # collect documents per query
    queries_result_list = defaultdict(dict)
    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute similarities
            cos_scores = dot_score(query_embeddings[query_start_idx:query_start_idx+query_chunk_size], corpus_embeddings[corpus_start_idx:corpus_start_idx+corpus_chunk_size])
            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id][corpus_id] = score
    # sort and strip to top_k results; convert to original indices
    pairs = set()
    for mention_idx, entity_scores in queries_result_list.items():
        top_k_entity_scores = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        pairs.update({Pair(mention_ids[mention_idx], entity_ids[entity_idx], score) for entity_idx, score in top_k_entity_scores})
    return pairs
