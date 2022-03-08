from typing import Tuple, List, Optional
from spacy.lang.en import English
from collections import defaultdict
import impl.dbpedia.util as dbp_util
import impl.util.spacy.listing_parser as list_nlp
import impl.listpage.util as list_util
import impl.wikipedia.wikimarkup_parser as wmp
from tqdm import tqdm
from enum import Enum
import multiprocessing as mp
import utils


class BertSpecialToken(Enum):
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


class WordTokenizer:
    """Takes parsed wiki markup and splits it into word tokens while preserving entity labels."""
    def __init__(self, max_words_per_chunk=384):
        self.word_tokenizer = English().tokenizer
        self.max_words_per_chunk = max_words_per_chunk
        self.meta_sections = {'see also', 'external links', 'references', 'notes'}

    def __call__(self, pages: dict, entity_labels=None) -> dict:
        if entity_labels:
            def page_with_labels_iterator():
                for page_uri, page_data in pages.items():
                    yield page_uri, page_data, entity_labels[page_uri]
            page_items = tqdm(page_with_labels_iterator(), total=len(pages), desc='Tokenize Pages (train)')
            tokenize_fn = self._tokenize_with_entities
        else:
            page_items = tqdm(pages.items(), total=len(pages), desc='Tokenize Pages (all)')
            tokenize_fn = self._tokenize

        with mp.Pool(processes=utils.get_config('max_cpus')) as pool:
            return {page_uri: tokens for page_uri, tokens in pool.imap_unordered(tokenize_fn, page_items, chunksize=2000) if tokens[0]}

    def _tokenize_with_entities(self, params) -> Tuple[str, Tuple[list, list]]:
        """Take a page and return list of (tokens, entities) chunks. If not existing, entity for a token is None."""
        page_uri, page_data, (valid_ents, invalid_ents) = params
        page_name = list_util.listpage2name(page_uri)

        page_tokens, page_ents = [], []
        if not valid_ents | invalid_ents:
            # can't produce entity labels without valid/invalid entities, so we discard the page
            return page_uri, (page_tokens, page_ents)

        top_section_name = ''
        for section_data in page_data['sections']:
            section_name = section_data['name']
            top_section_name = section_name if section_data['level'] <= 2 else top_section_name
            if top_section_name.lower() in self.meta_sections:
                continue  # skip meta sections

            for enum_data in section_data['enums']:
                context_tokens, _ = self._context_to_tokens([page_name, top_section_name, section_name])
                context_ents = [-100] * len(context_tokens)

                max_chunk_size = self.max_words_per_chunk - len(context_tokens)
                for chunk_tokens, chunk_ents in self._listing_to_token_entity_chunks(enum_data, valid_ents, invalid_ents, max_chunk_size):
                    page_tokens.append(context_tokens + chunk_tokens)
                    page_ents.append(context_ents + chunk_ents)

            for table in section_data['tables']:
                table_header, table_data = table['header'], table['data']
                context_tokens, _ = self._context_to_tokens([page_name, top_section_name, section_name], table_header)
                context_ents = [-100] * len(context_tokens)

                max_chunk_size = self.max_words_per_chunk - len(context_tokens)
                for chunk_tokens, chunk_ents in self._listing_to_token_entity_chunks(table_data, valid_ents, invalid_ents, max_chunk_size):
                    page_tokens.append(context_tokens + chunk_tokens)
                    page_ents.append(context_ents + chunk_ents)
        return page_uri, (page_tokens, page_ents)

    def _context_to_tokens(self, context: List[str], table_header=None) -> Tuple[list, list]:
        """Converts list of context strings (and table header) to context tokens."""
        ctx_tokens, ctx_ws = [], []

        # add listing context, separated by special context tokens
        for text in context:
            doc = self.word_tokenizer(wmp.wikitext_to_plaintext(text))
            ctx_tokens.extend([w.text for w in doc] + [BertSpecialToken.CONTEXT_SEP.value])
            ctx_ws.extend([w.whitespace_ for w in doc] + [' '])

        # add table header if available
        if table_header:
            for cell in table_header:
                doc = self.word_tokenizer(cell['text'])
                ctx_tokens.extend([w.text for w in doc] + [BertSpecialToken.TABLE_COL.value])
                ctx_ws.extend([w.whitespace_ for w in doc] + [' '])

        ctx_tokens[-1] = BertSpecialToken.CONTEXT_END.value  # replace last token with final context separator
        return ctx_tokens, ctx_ws

    def _listing_to_token_entity_chunks(self, listing_data: list, valid_ents: set, invalid_ents: set, max_group_size: int):
        """Converts a listing to a set of (tokens, entities) chunks."""
        current_group_size = 0
        current_group_tokens = []
        current_group_ents = []
        for item in listing_data:
            # pick conversion function based on item being a list of table cells or an enum entry string
            item_to_tokens = self._row_to_tokens_and_entities if type(item) == list else self._entry_to_tokens_and_entities
            item_tokens, item_ents = item_to_tokens(item, valid_ents, invalid_ents)

            if not item_tokens:
                continue  # skip if no labeled entities are found
            item_size = len(item_tokens)
            new_group_size = current_group_size + item_size
            if not current_group_tokens or new_group_size <= max_group_size:
                current_group_tokens.extend(item_tokens)
                current_group_ents.extend(item_ents)
                current_group_size = new_group_size
            else:
                yield current_group_tokens, current_group_ents
                current_group_tokens = item_tokens
                current_group_ents = item_ents
                current_group_size = item_size
        if current_group_tokens:
            yield current_group_tokens, current_group_ents

    def _entry_to_tokens_and_entities(self, entry: dict, valid_ents: set, invalid_ents: set) -> Tuple[list, list]:
        entry_text = entry['text']
        entry_doc = list_nlp.parse(entry_text)
        depth = entry['depth']

        # check whether entry is valid training data
        entities = list(entry['entities'])
        entities.extend(self._find_untagged_entities(entry_doc, entities))
        entity_names = {e['name'] for e in entities}
        if all(en not in valid_ents for en in entity_names) and any(en not in invalid_ents for en in entity_names):
            # discard entry if 1) there are no valid entities we know of and
            # 2) we are unsure about other entities (i.e., they MAY be valid entities)
            return [], []

        # extract tokens and entities
        tokens, _, ents = self._text_to_tokens(entry_doc, entities, valid_ents)
        if not tokens or not ents:
            return [], []
        return [BertSpecialToken.get_entry_by_depth(depth)] + tokens, [None] + ents

    def _row_to_tokens_and_entities(self, row: list, valid_ents: set, invalid_ents: set) -> Tuple[list, list]:
        cell_docs = [list_nlp.parse(cell['text']) for cell in row]
        # check whether row is valid training data
        row_entities = []
        for cell_idx, cell in enumerate(row):
            cell_doc = cell_docs[cell_idx]
            cell_entities = list(cell['entities'])
            row_entities.extend(cell_entities)
            row_entities.extend(self._find_untagged_entities(cell_doc, cell_entities))
        entity_names = {e['name'] for e in row_entities}
        if all(en not in valid_ents for en in entity_names) and any(en not in invalid_ents for en in entity_names):
            # discard entry if 1) there are no valid entities we know of and
            # 2) we are unsure about other entities (i.e., they MAY be valid entities)
            return [], []

        # extract tokens and entities
        tokens, ents = [], []
        for cell_idx, cell in enumerate(row):
            cell_doc = cell_docs[cell_idx]
            cell_entities = list(cell['entities'])
            cell_tokens, _, cell_ents = self._text_to_tokens(cell_doc, cell_entities, valid_ents)
            tokens += [BertSpecialToken.TABLE_COL.value] + cell_tokens
            ents += [None] + cell_ents
        if tokens:
            tokens[0] = BertSpecialToken.TABLE_ROW.value  # special indicator for start of table row
        return tokens, ents

    def _find_untagged_entities(self, doc, entities: list):
        """Adds a dummy entity to the list of entities of a document if there is at least one unidentified entity."""
        untagged_entities = []

        entity_character_idxs = set()
        for entity_data in entities:
            start = entity_data['idx']
            end = start + len(entity_data['text'])
            entity_character_idxs.update(range(start, end))
        for ent in doc.ents:
            start = ent.start_char
            end = ent.end_char
            text = ent.text
            while start in entity_character_idxs and start < end:
                start += 1
                text = text[1:]
            while (end - 1) in entity_character_idxs and start < end:
                end -= 1
                text = text[:-1]
            if not entity_character_idxs.intersection(set(range(start, end))) and len(text.strip()) > 1:
                untagged_entities.append({'name': 'UNTAGGED-ENT'})
                break  # finding one is already enough
        return untagged_entities

    def _tokenize(self, params) -> Tuple[str, Tuple[list, list]]:
        """Takes a page and returns list of token chunks."""
        page_uri, page_data = params
        is_list_page = list_util.is_listpage(page_uri)
        page_name = list_util.listpage2name(page_uri) if is_list_page else dbp_util.resource2name(page_uri)
        page_tokens, page_ws = [], []

        top_section_name = ''
        for section_data in page_data['sections']:
            section_name = section_data['name']
            top_section_name = section_name if section_data['level'] <= 2 else top_section_name
            if top_section_name in self.meta_sections:
                continue  # skip meta sections
            for enum_data in section_data['enums']:
                context_tokens, context_ws = self._context_to_tokens([page_name, top_section_name, section_name])
                max_group_size = self.max_words_per_chunk - len(context_tokens)
                for group_tokens, group_ws in self._listing_to_token_groups(enum_data, max_group_size):
                    page_tokens.append(context_tokens + group_tokens)
                    page_ws.append(context_ws + group_ws)
            for table in section_data['tables']:
                table_header, table_data = table['header'], table['data']
                context_tokens, context_ws = self._context_to_tokens([page_name, top_section_name, section_name], table_header)
                max_group_size = self.max_words_per_chunk - len(context_tokens)
                for group_tokens, group_ws in self._listing_to_token_groups(table_data, max_group_size):
                    page_tokens.append(context_tokens + group_tokens)
                    page_ws.append(context_ws + group_ws)
        return page_uri, (page_tokens, page_ws)

    def _listing_to_token_groups(self, listing_data, max_group_size: int):
        current_group_size = 0
        current_group_tokens = []
        current_group_ws = []
        for item in listing_data:
            # pick conversion function based on item being a list of table cells or an enum entry dict
            item_to_tokens = self._row_to_tokens if type(item) == list else self._entry_to_tokens
            item_tokens, item_ws = item_to_tokens(item)

            if not item_tokens:
                continue  # skip if no tokens are found
            item_size = len(item_tokens)
            new_group_size = current_group_size + item_size
            if not current_group_tokens or new_group_size <= max_group_size:
                current_group_tokens.extend(item_tokens)
                current_group_ws.extend(item_ws)
                current_group_size = new_group_size
            else:
                yield current_group_tokens, current_group_ws
                current_group_tokens = item_tokens
                current_group_ws = item_ws
                current_group_size = item_size
        if current_group_tokens:
            yield current_group_tokens, current_group_ws

    def _entry_to_tokens(self, entry: dict) -> Tuple[list, list]:
        entry_doc = self.word_tokenizer(entry['text'])
        depth = entry['depth']
        tokens, ws, _ = self._text_to_tokens(entry_doc, None, None)
        return [BertSpecialToken.get_entry_by_depth(depth)] + tokens, [' '] + ws

    def _row_to_tokens(self, row: list) -> Tuple[list, list]:
        tokens, ws = [], []
        for cell in row:
            cell_tokens, cell_ws, _ = self._text_to_tokens(self.word_tokenizer(cell['text']), None, None)
            tokens += [BertSpecialToken.TABLE_COL.value] + cell_tokens
            ws += [' '] + cell_ws
        if tokens:
            tokens[0] = BertSpecialToken.TABLE_ROW.value
        return tokens, ws

    def _text_to_tokens(self, doc, entities: Optional[list], valid_ents: Optional[set]) -> Tuple[list, list, list]:
        """Transforms a spacy doc (and entity info) to lists of tokens, whitespaces(, and entities)."""
        if not entities or not valid_ents:
            tokens = [w.text for w in doc]
            ws = [w.whitespace_ for w in doc]
            return tokens, ws, [None] * len(tokens)

        tokens, ws, token_ents = [], [], []
        entity_pos_map = self._create_entity_position_map(entities, valid_ents)
        current_position = 0
        for w in doc:
            tokens.append(w.text)
            ws.append(w.whitespace_)
            token_ents.append(entity_pos_map[current_position])
            current_position += len(w.text_with_ws)
        return tokens, ws, token_ents

    def _create_entity_position_map(self, entities: list, valid_ents: set):
        """Index valid entities by their text position."""
        entity_pos_map = defaultdict(lambda: None)
        filtered_ents = [e for e in entities if e['name'] in valid_ents]
        for ent in filtered_ents:
            idx = ent['idx']
            text_length = len(ent['text'])
            for i in range(idx, idx + text_length):
                entity_pos_map[i] = ent['name']
        return entity_pos_map
