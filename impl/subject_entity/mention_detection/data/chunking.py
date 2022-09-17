from typing import List, Tuple, Dict, Union, Optional, Iterable
from collections import namedtuple, defaultdict
import random
from copy import copy
import multiprocessing as mp
import utils
from tqdm import tqdm
from impl.util.spacy import get_tokens_and_whitespaces_from_text
from impl.util.transformer import SpecialToken, EntityIndex
from impl.wikipedia import WikiPage
from impl.wikipedia.page_parser import WikiListing, WikiEnum, WikiTable, WikiListingItem, WikiEnumEntry, WikiTableRow

MAX_TOKENS_PER_CHUNK = 300
MIN_ITEMS_PER_CHUNK = 3
MAX_ITEMS_PER_CHUNK = 16
MAX_TOKENS_PER_ITEM = 30

TransformerItem = namedtuple('TransformerItem', ['idx', 'tokens', 'whitespaces', 'labels'])


def process_training_pages(pages: List[WikiPage], labels: Dict[int, Dict[int, Dict[int, List[Union[int, List[int]]]]]], negative_sample_size: float) -> Tuple[List[List[str]], List[List[int]], List[str]]:
    token_chunks, label_chunks, types_of_chunks = [], [], []
    listings = [listing for p in pages for listing in p.get_listings()]
    for listing, listing_tokens, _, listing_labels, _ in _chunk_listings(listings, labels):
        token_chunks.extend(listing_tokens)
        label_chunks.extend(listing_labels)
        types_of_chunks.extend([listing.get_type()] * len(listing_tokens))
    if negative_sample_size > 0:
        negative_listings = _create_negative_listings(listings, types_of_chunks, negative_sample_size)
        for listing, listing_tokens, _, listing_labels, _ in _chunk_listings(negative_listings, None):
            token_chunks.extend(listing_tokens)
            label_chunks.extend(listing_labels)
            types_of_chunks.extend([listing.get_type()] * len(listing_tokens))
    return token_chunks, label_chunks, types_of_chunks


def _create_negative_listings(listings: List[WikiListing], types_of_chunks: List[str], negative_sample_size: float) -> List[WikiListing]:
    # compute number of negative chunks to create
    num_enum_chunks = len([t for t in types_of_chunks if t == WikiEnum.get_type()])
    num_table_chunks = len(types_of_chunks) - num_enum_chunks
    num_negative_enum_chunks = int(num_enum_chunks * negative_sample_size)
    num_negative_table_chunks = int(num_table_chunks * negative_sample_size)
    enum_listings = [listing for listing in listings if isinstance(listing, WikiEnum)]
    table_listings = defaultdict(list)  # group table listings by columns (to make the sample more realistic)
    for listing in listings:
        if isinstance(listing, WikiTable):
            table_listings[len(listing.header.tokens)].append(listing)
    # generate negatives
    negative_listings = []
    for _ in range(num_negative_enum_chunks):
        negative_listings.append(_create_negative_listing(enum_listings))
    for _ in range(num_negative_table_chunks):
        # pick number of table columns for negative listing (weighted by relative frequency of columns)
        num_cols = random.choices(list(table_listings), weights=[len(vals) for vals in table_listings.values()], k=1)[0]
        negative_listings.append(_create_negative_listing(table_listings[num_cols]))
    return negative_listings


def _create_negative_listing(listings: List[WikiListing]) -> WikiListing:
    # select listing context (i.e. pick a random listing)
    negative_listing = copy(random.choice(listings))
    # randomly assign items from arbitrary listings
    num_items = random.choice(range(MIN_ITEMS_PER_CHUNK, MAX_ITEMS_PER_CHUNK + 1))
    items = [random.choice(list(listing.get_items())) for listing in random.sample(listings, num_items)]
    negative_listing.items = {idx: item for idx, item in enumerate(items)}
    return negative_listing


def process_pages(pages: List[WikiPage]) -> Tuple[List[List[Tuple[int, int, int]]], List[List[str]], List[List[str]]]:
    context_chunks, token_chunks, whitespace_chunks = [], [], []
    listings = [listing for page in pages for listing in page.get_listings()]
    for listing, listing_tokens, listing_whitespaces, _, listing_items in _chunk_listings(listings, None):
        token_chunks.extend(listing_tokens)
        whitespace_chunks.extend(listing_whitespaces)
        context_chunks.extend([[(listing.page.idx, listing.idx, item_idx) for item_idx in items] for items in listing_items])
    return context_chunks, token_chunks, whitespace_chunks


def _chunk_listings(listings: List[WikiListing], labels: Optional[Dict[int, Dict[int, Dict[int, List[Union[int, List[int]]]]]]]) -> Iterable[Tuple[WikiListing, List[List[str]], List[List[str]], List[List[int]], List[List[int]]]]:
    if labels is None:
        listings_with_labels = [(listing, None) for listing in listings]
    else:
        listings_with_labels = [(listing, labels[listing.page.idx][listing.idx]) for listing in listings if listing.page.idx in labels and listing.idx in labels[listing.page.idx]]
    with mp.Pool(processes=(utils.get_config('max_cpus') // 3)) as pool:
        yield from pool.imap_unordered(_chunk_listing, tqdm(listings_with_labels, desc='Chunking listings'), chunksize=1000)


def _chunk_listing(listing_and_labels: Tuple[WikiListing, Optional[Dict[int, List[Union[int, List[int]]]]]]) -> Tuple[WikiListing, List[List[str]], List[List[str]], List[List[int]], List[List[int]]]:
    listing, labels = listing_and_labels
    listing_token_chunks, listing_whitespace_chunks, listing_label_chunks, listing_chunk_items = [], [], [], []
    listing_context = _process_listing_context(listing)

    max_chunk_size = MAX_TOKENS_PER_CHUNK - len(listing_context.tokens)
    current_chunk_size = 0
    items_per_chunk = []
    for item in listing.get_items():
        if labels and item.idx not in labels:
            continue
        processed_item = _process_listing_item(item, labels)
        if processed_item is None:
            continue
        new_chunk_size = current_chunk_size + len(processed_item.tokens)
        if not items_per_chunk or new_chunk_size > max_chunk_size or len(items_per_chunk[-1]) >= MAX_ITEMS_PER_CHUNK:
            items_per_chunk.append([listing_context, processed_item])
            current_chunk_size = len(processed_item.tokens)
        else:
            items_per_chunk[-1].append(processed_item)
            current_chunk_size = new_chunk_size
    # convert to lists of tokens
    for items in items_per_chunk:
        if len(items) <= MIN_ITEMS_PER_CHUNK:
            continue
        listing_token_chunks.append([t for i in items for t in i.tokens])
        listing_whitespace_chunks.append([ws for i in items for ws in i.whitespaces])
        listing_label_chunks.append([label for i in items for label in i.labels])
        listing_chunk_items.append([i.idx for i in items if i.idx is not None])
    return listing, listing_token_chunks, listing_whitespace_chunks, listing_label_chunks, listing_chunk_items


def _process_listing_context(listing: WikiListing) -> TransformerItem:
    ctx_tokens, ctx_whitespaces = [], []
    # add page context
    page_tokens, page_whitespaces = get_tokens_and_whitespaces_from_text(listing.page.resource.get_label())
    ctx_tokens.extend(page_tokens + [SpecialToken.CONTEXT_SEP.value])
    ctx_whitespaces.extend(page_whitespaces + [' '])
    # add topsection and section
    ctx_tokens.extend(listing.topsection.tokens + [SpecialToken.CONTEXT_SEP.value])
    ctx_whitespaces.extend(listing.topsection.whitespaces + [' '])
    ctx_tokens.extend(listing.section.tokens + [SpecialToken.CONTEXT_SEP.value])
    ctx_whitespaces.extend(listing.section.whitespaces + [' '])
    # add table header if available
    if isinstance(listing, WikiTable):
        for cell_tokens, cell_whitespaces in zip(listing.header.tokens, listing.header.whitespaces):
            ctx_tokens.extend(cell_tokens + [SpecialToken.TABLE_COL.value])
            ctx_whitespaces.extend(cell_whitespaces + [' '])
    ctx_tokens[-1] = SpecialToken.CONTEXT_END.value  # replace last token with final context separator
    ctx_labels = [EntityIndex.IGNORE.value] * len(ctx_tokens)
    return TransformerItem(idx=None, tokens=ctx_tokens, whitespaces=ctx_whitespaces, labels=ctx_labels)


def _process_listing_item(item: WikiListingItem, labels: Optional[Dict[int, List[Union[int, List[int]]]]]) -> Optional[TransformerItem]:
    item_labels = labels[item.idx] if labels else None
    if isinstance(item, WikiEnumEntry):
        if not item.tokens:
            return None
        if item_labels is None:
            item_labels = [EntityIndex.NO_ENTITY.value] * len(item.tokens)
        tokens = [SpecialToken.get_entry_by_depth(item.depth)] + item.tokens
        whitespaces = [' '] + item.whitespaces
        labels = [EntityIndex.NO_ENTITY.value] + item_labels
        return TransformerItem(
            idx=item.idx,
            tokens=tokens[:MAX_TOKENS_PER_ITEM],
            whitespaces=whitespaces[:MAX_TOKENS_PER_ITEM],
            labels=labels[:MAX_TOKENS_PER_ITEM]
        )
    elif isinstance(item, WikiTableRow):  # WikiTableRow
        if item_labels is None:
            item_labels = [[EntityIndex.NO_ENTITY.value] * len(cell_tokens) for cell_tokens in item.tokens]
        tokens, whitespaces, labels = [], [], []
        for cell_tokens, cell_whitespaces, cell_labels in zip(item.tokens, item.whitespaces, item_labels):
            tokens += [SpecialToken.TABLE_COL.value] + cell_tokens
            whitespaces += [' '] + cell_whitespaces
            labels += [EntityIndex.NO_ENTITY.value] + cell_labels
        if not tokens:
            return None
        tokens[0] = SpecialToken.TABLE_ROW.value  # special indicator for start of table row
        return TransformerItem(
            idx=item.idx,
            tokens=tokens[:MAX_TOKENS_PER_ITEM],
            whitespaces=whitespaces[:MAX_TOKENS_PER_ITEM],
            labels=labels[:MAX_TOKENS_PER_ITEM]
        )
    else:
        raise ValueError(f'Can not process item: {item}')
