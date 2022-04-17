from typing import Tuple, List, Optional, Set, Dict
from spacy.lang.en import English
from collections import defaultdict
import impl.util.spacy.listing_parser as list_nlp
import impl.wikipedia.wikimarkup_parser as wmp
from tqdm import tqdm
from enum import Enum
import multiprocessing as mp
from impl.dbpedia.resource import DbpResource


class WordTokenizerSpecialToken(Enum):
    CONTEXT_SEP = '[CXS]'
    CONTEXT_END = '[CXE]'
    TABLE_ROW = '[ROW]'
    TABLE_COL = '[COL]'
    ENTRY_L1 = '[E1]'
    ENTRY_L2 = '[E2]'
    ENTRY_L3 = '[E3]'

    @classmethod
    def all_tokens(cls):
        return {t.value for t in cls}

    @classmethod
    def item_starttokens(cls):
        return {cls.TABLE_ROW.value, cls.ENTRY_L1.value, cls.ENTRY_L2.value, cls.ENTRY_L3.value}

    @classmethod
    def get_entry_by_depth(cls, depth: int):
        if depth == 1:
            return cls.ENTRY_L1.value
        elif depth == 2:
            return cls.ENTRY_L2.value
        elif depth >= 3:
            return cls.ENTRY_L3.value
        raise ValueError(f'Trying to retrieve a BERT special token for an entry of depth {depth}.')


class WordTokenizerSpecialLabel(Enum):
    NEW_ENTITY = -1
    NO_ENTITY = -2
    IGNORE = -100


class WordTokenizer:
    """Takes parsed wiki markup and splits it into word tokens while preserving entity labels."""
    def __init__(self, max_words_per_chunk=384):
        self.word_tokenizer = English().tokenizer
        self.max_words_per_chunk = max_words_per_chunk
        self.meta_sections = {'see also', 'external links', 'references', 'notes'}

    def __call__(self, pages: Dict[DbpResource, dict], entity_labels: Dict[DbpResource, Tuple[Set[int], Set[int]]] = None) -> Dict[DbpResource, Tuple[list, list]]:
        if entity_labels:
            def page_with_labels_iterator():
                for res, page_data in pages.items():
                    yield res, page_data, entity_labels[res]
            page_items = tqdm(page_with_labels_iterator(), total=len(pages), desc='Tokenize Pages (train)')
            tokenize_fn = self._tokenize_page_with_entities
        else:
            page_items = tqdm(pages.items(), total=len(pages), desc='Tokenize Pages (all)')
            tokenize_fn = self._tokenize_page

        with mp.Pool(processes=2) as pool:
            return {res: tokens for res, tokens in pool.imap_unordered(tokenize_fn, page_items, chunksize=1000) if tokens[0]}

    def _tokenize_page_with_entities(self, params: Tuple[DbpResource, dict, Tuple[Set[int], Set[int]]]) -> Tuple[DbpResource, Tuple[list, list]]:
        """Take a resource and return list of (tokens, entities) chunks. If not existing, entity for a token is None."""
        res, page_data, (valid_ents, invalid_ents) = params

        page_tokens, page_ents = [], []
        if not valid_ents | invalid_ents:
            # can't produce entity labels without valid/invalid entities, so we discard the page
            return res, (page_tokens, page_ents)

        top_section_name = ''
        for section_data in page_data['sections']:
            section_name = section_data['name']
            top_section_name = section_name if section_data['level'] <= 2 else top_section_name
            if top_section_name.lower() in self.meta_sections:
                continue  # skip meta sections
            for enum_data in section_data['enums']:
                context_tokens, _ = self._context_to_tokens([res.get_label(), top_section_name, section_name])
                context_ents = [WordTokenizerSpecialLabel.IGNORE.value] * len(context_tokens)
                max_chunk_size = self.max_words_per_chunk - len(context_tokens)
                for chunk_tokens, chunk_ents in self._listing_to_token_entity_chunks(enum_data, valid_ents, invalid_ents, max_chunk_size):
                    page_tokens.append(context_tokens + chunk_tokens)
                    page_ents.append(context_ents + chunk_ents)
            for table in section_data['tables']:
                table_header, table_data = table['header'], table['data']
                context_tokens, _ = self._context_to_tokens([res.get_label(), top_section_name, section_name], table_header)
                context_ents = [WordTokenizerSpecialLabel.IGNORE.value] * len(context_tokens)
                max_chunk_size = self.max_words_per_chunk - len(context_tokens)
                for chunk_tokens, chunk_ents in self._listing_to_token_entity_chunks(table_data, valid_ents, invalid_ents, max_chunk_size):
                    page_tokens.append(context_tokens + chunk_tokens)
                    page_ents.append(context_ents + chunk_ents)
        return res, (page_tokens, page_ents)

    def _context_to_tokens(self, context: List[str], table_header=None) -> Tuple[list, list]:
        """Converts list of context strings (and table header) to context tokens."""
        ctx_tokens, ctx_ws = [], []

        # add listing context, separated by special context tokens
        for text in context:
            doc = self.word_tokenizer(wmp.wikitext_to_plaintext(text))
            ctx_tokens.extend([w.text for w in doc] + [WordTokenizerSpecialToken.CONTEXT_SEP.value])
            ctx_ws.extend([w.whitespace_ for w in doc] + [' '])

        # add table header if available
        if table_header:
            for cell in table_header:
                doc = self.word_tokenizer(cell['text'])
                ctx_tokens.extend([w.text for w in doc] + [WordTokenizerSpecialToken.TABLE_COL.value])
                ctx_ws.extend([w.whitespace_ for w in doc] + [' '])

        ctx_tokens[-1] = WordTokenizerSpecialToken.CONTEXT_END.value  # replace last token with final context separator
        return ctx_tokens, ctx_ws

    def _listing_to_token_entity_chunks(self, listing_data: list, valid_ents: Set[int], invalid_ents: Set[int], max_chunk_size: int):
        """Converts a listing to a set of (tokens, entities) chunks."""
        current_chunk_size = 0
        current_chunk_tokens = []
        current_chunk_ents = []
        for item in listing_data:
            # pick conversion function based on item being a list of table cells or an enum entry string
            item_to_tokens = self._row_to_tokens_and_entities if isinstance(item, list) else self._entry_to_tokens_and_entities
            item_tokens, item_ents = item_to_tokens(item, valid_ents, invalid_ents)

            if not item_tokens:
                continue  # skip if no labeled entities are found
            item_size = len(item_tokens)
            new_chunk_size = current_chunk_size + item_size
            if not current_chunk_tokens or new_chunk_size <= max_chunk_size:
                current_chunk_tokens.extend(item_tokens)
                current_chunk_ents.extend(item_ents)
                current_chunk_size = new_chunk_size
            else:
                yield current_chunk_tokens, current_chunk_ents
                current_chunk_tokens = item_tokens
                current_chunk_ents = item_ents
                current_chunk_size = item_size
        if current_chunk_tokens:
            yield current_chunk_tokens, current_chunk_ents

    def _entry_to_tokens_and_entities(self, entry: dict, valid_ents: Set[int], invalid_ents: Set[int]) -> Tuple[list, list]:
        entry_text = entry['text']
        entry_entities = entry['entities']
        entry_doc = list_nlp.parse(entry_text)
        depth = entry['depth']

        # check whether entry is valid training data
        entity_indices = {e['idx'] for e in entry_entities}
        has_valid_entities = len(entity_indices.intersection(valid_ents)) > 0
        has_untagged_entities = self._has_untagged_entities(entry_doc, entry_entities)
        has_unclear_entities = has_untagged_entities or not entity_indices.issubset(invalid_ents)
        if not has_valid_entities and has_unclear_entities:
            # discard entry if 1) there are no valid entities we know of and
            # 2) there are some entities about which we don't know anything (i.e., they MAY be valid entities)
            return [], []

        # extract tokens and entities
        tokens, _, ents = self._text_to_tokens(entry_doc, entry_entities, valid_ents)
        if not tokens or not ents:
            return [], []
        return [WordTokenizerSpecialToken.get_entry_by_depth(depth)] + tokens, [WordTokenizerSpecialLabel.NO_ENTITY.value] + ents

    def _row_to_tokens_and_entities(self, row: list, valid_ents: Set[int], invalid_ents: Set[int]) -> Tuple[list, list]:
        cell_docs = [list_nlp.parse(cell['text']) for cell in row]
        # check whether row is valid training data
        row_entity_indices = set()
        has_untagged_entities = False
        for cell_idx, cell in enumerate(row):
            cell_entities = list(cell['entities'])
            row_entity_indices.update({e['idx'] for e in cell_entities})
            has_untagged_entities = has_untagged_entities or self._has_untagged_entities(cell_docs[cell_idx], cell_entities)
        has_valid_entities = len(row_entity_indices.intersection(valid_ents)) > 0
        has_unclear_entities = has_untagged_entities or not row_entity_indices.issubset(invalid_ents)
        if not has_valid_entities and has_unclear_entities:
            # discard entry if 1) there are no valid entities we know of and
            # 2) there are some entities about which we don't know anything (i.e., they MAY be valid entities)
            return [], []

        # extract tokens and entities
        tokens, ents = [], []
        for cell_idx, cell in enumerate(row):
            cell_tokens, _, cell_ents = self._text_to_tokens(cell_docs[cell_idx], cell['entities'], valid_ents)
            tokens += [WordTokenizerSpecialToken.TABLE_COL.value] + cell_tokens
            ents += [WordTokenizerSpecialLabel.NO_ENTITY.value] + cell_ents
        if tokens:
            tokens[0] = WordTokenizerSpecialToken.TABLE_ROW.value  # special indicator for start of table row
        return tokens, ents

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

    def _tokenize_page(self, params: Tuple[DbpResource, dict]) -> Tuple[DbpResource, Tuple[list, list]]:
        """Takes a resource and returns list of token chunks."""
        res, page_data = params
        page_tokens, page_ws = [], []

        top_section_name = ''
        for section_data in page_data['sections']:
            section_name = section_data['name']
            top_section_name = section_name if section_data['level'] <= 2 else top_section_name
            if top_section_name.lower() in self.meta_sections:
                continue  # skip meta sections
            for enum_data in section_data['enums']:
                context_tokens, context_ws = self._context_to_tokens([res.get_label(), top_section_name, section_name])
                max_chunk_size = self.max_words_per_chunk - len(context_tokens)
                for chunk_tokens, chunk_ws in self._listing_to_token_chunks(enum_data, max_chunk_size):
                    page_tokens.append(context_tokens + chunk_tokens)
                    page_ws.append(context_ws + chunk_ws)
            for table in section_data['tables']:
                table_header, table_data = table['header'], table['data']
                context_tokens, context_ws = self._context_to_tokens([res.get_label(), top_section_name, section_name], table_header)
                max_chunk_size = self.max_words_per_chunk - len(context_tokens)
                for chunk_tokens, chunk_ws in self._listing_to_token_chunks(table_data, max_chunk_size):
                    page_tokens.append(context_tokens + chunk_tokens)
                    page_ws.append(context_ws + chunk_ws)
        return res, (page_tokens, page_ws)

    def _listing_to_token_chunks(self, listing_data: list, max_chunk_size: int):
        current_chunk_size = 0
        current_chunk_tokens = []
        current_chunk_ws = []
        for item in listing_data:
            # pick conversion function based on item being a list of table cells or an enum entry dict
            item_to_tokens = self._row_to_tokens if isinstance(item, list) else self._entry_to_tokens
            item_tokens, item_ws = item_to_tokens(item)

            if not item_tokens:
                continue  # skip if no tokens are found
            item_size = len(item_tokens)
            new_chunk_size = current_chunk_size + item_size
            if not current_chunk_tokens or new_chunk_size <= max_chunk_size:
                current_chunk_tokens.extend(item_tokens)
                current_chunk_ws.extend(item_ws)
                current_chunk_size = new_chunk_size
            else:
                yield current_chunk_tokens, current_chunk_ws
                current_chunk_tokens = item_tokens
                current_chunk_ws = item_ws
                current_chunk_size = item_size
        if current_chunk_tokens:
            yield current_chunk_tokens, current_chunk_ws

    def _entry_to_tokens(self, entry: dict) -> Tuple[list, list]:
        entry_doc = self.word_tokenizer(entry['text'])
        depth = entry['depth']
        tokens, ws, _ = self._text_to_tokens(entry_doc, None, None)
        return [WordTokenizerSpecialToken.get_entry_by_depth(depth)] + tokens, [' '] + ws

    def _row_to_tokens(self, row: list) -> Tuple[list, list]:
        tokens, ws = [], []
        for cell in row:
            cell_tokens, cell_ws, _ = self._text_to_tokens(self.word_tokenizer(cell['text']), None, None)
            tokens += [WordTokenizerSpecialToken.TABLE_COL.value] + cell_tokens
            ws += [' '] + cell_ws
        if tokens:
            tokens[0] = WordTokenizerSpecialToken.TABLE_ROW.value
        return tokens, ws

    def _text_to_tokens(self, doc, entities: Optional[list], valid_ents: Optional[Set[int]]) -> Tuple[list, list, list]:
        """Transforms a spacy doc (and entity info) to lists of tokens, whitespaces(, and entities)."""
        if not entities or not valid_ents:
            tokens = [w.text for w in doc]
            ws = [w.whitespace_ for w in doc]
            return tokens, ws, [WordTokenizerSpecialLabel.NO_ENTITY.value] * len(tokens)

        tokens, ws, token_ents = [], [], []
        entity_pos_map = self._create_entity_position_map(entities, valid_ents)
        current_position = 0
        for w in doc:
            tokens.append(w.text)
            ws.append(w.whitespace_)
            token_ents.append(entity_pos_map[current_position])
            current_position += len(w.text_with_ws)
        return tokens, ws, token_ents

    def _create_entity_position_map(self, entities: list, valid_ents: Set[int]):
        """Index valid entities by their text position."""
        entity_pos_map = defaultdict(lambda: WordTokenizerSpecialLabel.NO_ENTITY.value)
        for ent in entities:
            if ent['idx'] not in valid_ents:
                continue
            start = ent['start']
            text_length = len(ent['text'])
            for i in range(start, start + text_length):
                entity_pos_map[i] = ent['idx']
        return entity_pos_map
