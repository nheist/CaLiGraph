from typing import Tuple, List, Set, Dict, Optional
from tqdm import tqdm
from spacy.lang.en import English
from collections import defaultdict
from enum import Enum
import multiprocessing as mp
import utils
import impl.util.spacy.listing_parser as list_nlp
import impl.wikipedia.wikimarkup_parser as wmp
from impl.wikipedia import WikipediaPage, META_SECTIONS
from impl.util.transformer import EntityIndex, SpecialToken


class ListingType(Enum):
    ENUMERATION = 'enumeration'
    TABLE = 'table'


class WordTokenizedItem:
    tokens: List[str]
    whitespaces: List[str]
    entity_indices: List[int]

    def __init__(self, tokens: List[str], whitespaces: List[str], entity_indices: List[int]):
        self.tokens = tokens
        self.whitespaces = whitespaces
        self.entity_indices = entity_indices

    def __len__(self):
        return len(self.tokens)


class WordTokenizedListing:
    listing_type: ListingType
    topsection: str
    section: str
    column_count: Optional[int]  # only used for listing type `table`
    context: WordTokenizedItem
    items: List[WordTokenizedItem]

    def __init__(self, listing_type: ListingType, context: WordTokenizedItem, items: List[WordTokenizedItem], topsection: str, section: str, column_count=None):
        self.listing_type = listing_type
        self.context = context
        self.items = items
        self.topsection = topsection
        self.section = section
        self.column_count = column_count

    def to_chunks(self, max_items_per_chunk: int, max_words_per_chunk: int) -> Tuple[list, list, list]:
        listing_token_chunks, listing_ws_chunks, listing_label_chunks = [], [], []
        # group items into chunks with correct length
        max_chunk_size = max_words_per_chunk - len(self.context)
        current_chunk_size = 0
        items_per_chunk = []
        for i in self.items:
            if not items_per_chunk or current_chunk_size + len(i) > max_chunk_size or len(items_per_chunk[-1]) >= max_items_per_chunk:
                items_per_chunk.append([self.context, i])
                current_chunk_size = len(i)
            else:
                items_per_chunk[-1].append(i)
                current_chunk_size += len(i)
        # convert to lists of tokens
        for items in items_per_chunk:
            listing_token_chunks.append([t for i in items for t in i.tokens])
            listing_ws_chunks.append([ws for i in items for ws in i.whitespaces])
            listing_label_chunks.append([idx for i in items for idx in i.entity_indices])
        return listing_token_chunks, listing_ws_chunks, listing_label_chunks


class WordTokenizedPage:
    idx: int
    listings: List[WordTokenizedListing]

    def __init__(self, idx: int, listings: List[WordTokenizedListing]):
        self.idx = idx
        self.listings = listings

    def to_chunks(self, max_items_per_chunk: int, max_words_per_chunk: int) -> Tuple[list, list, list, list]:
        page_context_chunks, page_token_chunks, page_ws_chunks, page_label_chunks = [], [], [], []
        for l in self.listings:
            ctx = {'page_idx': self.idx, 'topsection': l.topsection, 'section': l.section, 'listing_type': l.listing_type}
            listing_token_chunks, listing_ws_chunks, listing_label_chunks = l.to_chunks(max_items_per_chunk, max_words_per_chunk)
            listing_context_chunks = [ctx] * len(listing_token_chunks)  # replicate context for all chunks of listing
            page_context_chunks.extend(listing_context_chunks)
            page_token_chunks.extend(listing_token_chunks)
            page_ws_chunks.extend(listing_ws_chunks)
            page_label_chunks.extend(listing_label_chunks)
        return page_context_chunks, page_token_chunks, page_ws_chunks, page_label_chunks


class WordTokenizer:
    """Takes parsed wiki markup and splits it into word tokens while preserving entity labels."""
    def __init__(self, max_ents_per_item: int = 999):
        self.max_ents_per_item = max_ents_per_item
        self.max_tokens_per_item = 30
        self.word_tokenizer = English().tokenizer

    def __call__(self, wiki_pages: List[WikipediaPage], entity_labels: Dict[int, Tuple[Set[int], Set[int]]] = None) -> List[WordTokenizedPage]:
        if entity_labels:
            n_processes = utils.get_config('max_cpus') // 2
            wiki_pages = {wp for wp in wiki_pages if entity_labels[wp.idx][0] | entity_labels[wp.idx][1]}  # discard pages without labels
        else:
            n_processes = 2
            entity_labels = defaultdict(lambda: (set(), set()))

        def page_with_labels_iterator():
            for wp in wiki_pages:
                yield wp, entity_labels[wp.idx]
        page_items = tqdm(page_with_labels_iterator(), total=len(wiki_pages), desc='Tokenize Pages')
        with mp.Pool(processes=n_processes) as pool:
            return [wtp for wtp in pool.imap_unordered(self._tokenize_page, page_items, chunksize=1000) if wtp.listings]

    def _tokenize_page(self, params: Tuple[WikipediaPage, Tuple[Set[int], Set[int]]]) -> WordTokenizedPage:
        """Take a resource and return list of (tokens, ws, entities) chunks."""
        wp, page_labels = params
        listings = []

        topsection = ''
        for section_data in wp.sections:
            section = section_data['name']
            topsection = section if section_data['level'] <= 2 else topsection
            if topsection.lower() in META_SECTIONS:
                continue  # skip meta sections
            for enum_data in section_data['enums']:
                ctx = self._tokenize_context([wp.resource.get_label(), topsection, section])
                items = [self._tokenize_entry(entry_data, page_labels) for entry_data in enum_data]
                items = [item for item in items if item is not None]  # get rid of invalid items
                if items:
                    listings.append(WordTokenizedListing(ListingType.ENUMERATION, ctx, items, topsection, section))
            for table in section_data['tables']:
                table_header, table_data = table['header'], table['data']
                ctx = self._tokenize_context([wp.resource.get_label(), topsection, section], table_header)
                items = [self._tokenize_row(row_data, page_labels) for row_data in table_data]
                items = [item for item in items if item is not None]  # get rid of invalid items
                if items:
                    listings.append(WordTokenizedListing(ListingType.TABLE, ctx, items, topsection, section, len(table_header)))
        return WordTokenizedPage(wp.idx, listings)

    def _tokenize_context(self, context: List[str], table_header=None) -> WordTokenizedItem:
        """Converts list of context strings (and table header) to context tokens."""
        ctx_tokens, ctx_ws = [], []
        # add listing context, separated by special context tokens
        for text in context:
            doc = self.word_tokenizer(wmp.wikitext_to_plaintext(text))
            ctx_tokens.extend([w.text for w in doc] + [SpecialToken.CONTEXT_SEP.value])
            ctx_ws.extend([w.whitespace_ for w in doc] + [' '])
        # add table header if available
        if table_header:
            for cell in table_header:
                doc = self.word_tokenizer(cell['text'])
                ctx_tokens.extend([w.text for w in doc] + [SpecialToken.TABLE_COL.value])
                ctx_ws.extend([w.whitespace_ for w in doc] + [' '])
        ctx_tokens[-1] = SpecialToken.CONTEXT_END.value  # replace last token with final context separator

        ctx_ents = [EntityIndex.IGNORE.value] * len(ctx_tokens)
        return WordTokenizedItem(ctx_tokens, ctx_ws, ctx_ents)

    def _tokenize_entry(self, entry: dict, page_labels: Tuple[set, set]) -> Optional[WordTokenizedItem]:
        entry_text = entry['text']
        depth = entry['depth']

        if page_labels[0] | page_labels[1]:  # page labels provided -> extract entities
            entry_doc = list_nlp.parse(entry_text)
            entry_entities = entry['entities']
            valid_ents, invalid_ents = page_labels
            # check whether entry is valid training data
            entity_indices = {e['idx'] for e in entry_entities}
            has_valid_entities = len(entity_indices.intersection(valid_ents)) > 0
            has_untagged_entities = self._has_untagged_entities(entry_doc, entry_entities)
            has_unclear_entities = has_untagged_entities or not entity_indices.issubset(invalid_ents)
            if not has_valid_entities and has_unclear_entities:
                # discard entry if 1) there are no valid entities we know of and
                # 2) there are some entities about which we don't know anything (i.e., they MAY be valid entities)
                return None
        else:  # only extract tokens and ignore entities
            entry_doc = self.word_tokenizer(entry_text)
            entry_entities = []
            valid_ents = set()

        entry_entities = [e for e in entry_entities if e['idx'] in valid_ents][:self.max_ents_per_item]
        tokens, ws, ents = self._text_to_tokens(entry_doc, entry_entities)
        tokens.insert(0, SpecialToken.get_entry_by_depth(depth))
        ws.insert(0, ' ')
        ents.insert(0, EntityIndex.NO_ENTITY.value)
        return WordTokenizedItem(tokens, ws, ents)

    def _tokenize_row(self, row: dict, page_labels: Tuple[set, set]) -> Optional[WordTokenizedItem]:
        if page_labels[0] | page_labels[1]:  # page labels provided -> extract entities
            cell_docs = [list_nlp.parse(cell['text']) for cell in row['cells']]
            valid_ents, invalid_ents = page_labels
            # check whether row is valid training data
            row_entity_indices = set()
            has_untagged_entities = False
            for cell_idx, cell in enumerate(row['cells']):
                row_entity_indices.update({e['idx'] for e in cell['entities']})
                has_untagged_entities = has_untagged_entities or self._has_untagged_entities(cell_docs[cell_idx], cell['entities'])
            has_valid_entities = len(row_entity_indices.intersection(valid_ents)) > 0
            has_unclear_entities = has_untagged_entities or not row_entity_indices.issubset(invalid_ents)
            if not has_valid_entities and has_unclear_entities:
                # discard entry if 1) there are no valid entities we know of and
                # 2) there are some entities about which we don't know anything (i.e., they MAY be valid entities)
                return None
        else:  # only extract tokens and ignore entities
            cell_docs = [self.word_tokenizer(cell['text']) for cell in row]
            valid_ents = set()

        tokens, ws, ents = [], [], []
        max_ents_per_row = self.max_ents_per_item
        for cell_idx, cell in enumerate(row):
            cell_ents = [e for e in cell['entities'] if e['idx'] in valid_ents][:max_ents_per_row]
            max_ents_per_row -= len(cell_ents)
            cell_tokens, cell_ws, cell_ents = self._text_to_tokens(cell_docs[cell_idx], cell_ents)
            tokens += [SpecialToken.TABLE_COL.value] + cell_tokens
            ws += [' '] + cell_ws
            ents += [EntityIndex.NO_ENTITY.value] + cell_ents
        if tokens:
            tokens[0] = SpecialToken.TABLE_ROW.value  # special indicator for start of table row
        return WordTokenizedItem(tokens, ws, ents)

    def _has_untagged_entities(self, doc, entities: list) -> bool:
        """True if there is at least one entity that is not tagged with wiki markup."""
        entity_character_idxs = set()
        for entity_data in entities:
            start = entity_data['start']
            end = start + len(entity_data['text'])
            entity_character_idxs.update(range(start, end))
        for ent in doc.ents:
            if not entity_character_idxs.intersection(set(range(ent.start_char, ent.end_char))):
                return True
        return False

    def _text_to_tokens(self, doc, entities: list) -> Tuple[list, list, list]:
        """Transforms a spacy doc and entity info to lists of tokens, whitespaces, and (potentially) entities."""
        doc = doc[:self.max_tokens_per_item]
        if not entities:
            tokens = [w.text for w in doc]
            ws = [w.whitespace_ for w in doc]
            return tokens, ws, [EntityIndex.NO_ENTITY.value] * len(tokens)

        tokens, ws, token_ents = [], [], []
        entity_pos_map = self._create_entity_position_map(entities)
        current_position = 0
        for w in doc:
            tokens.append(w.text)
            ws.append(w.whitespace_)
            token_ents.append(entity_pos_map[current_position])
            current_position += len(w.text_with_ws)
        return tokens, ws, token_ents

    def _create_entity_position_map(self, entities: list):
        """Index valid entities by their text position."""
        entity_pos_map = defaultdict(lambda: EntityIndex.NO_ENTITY.value)
        for ent in entities:
            start = ent['start']
            text_length = len(ent['text'])
            for i in range(start, start + text_length):
                entity_pos_map[i] = ent['idx']
        return entity_pos_map
