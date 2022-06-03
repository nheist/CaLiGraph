from typing import Tuple, List, Optional, Set, Dict
from tqdm import tqdm
from spacy.lang.en import English
from collections import defaultdict
from enum import Enum
import multiprocessing as mp
import utils
import impl.util.spacy.listing_parser as list_nlp
import impl.wikipedia.wikimarkup_parser as wmp
from impl.dbpedia.resource import DbpResource
from impl.util.rdf import EntityIndex


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
    NEW_ENTITY = EntityIndex.NEW_ENTITY.value
    NO_ENTITY = EntityIndex.NO_ENTITY.value
    IGNORE = -100


class WordTokenizer:
    """Takes parsed wiki markup and splits it into word tokens while preserving entity labels."""
    def __init__(self, max_words_per_chunk: int = 300, max_items_per_chunk: int = 16, max_ents_per_item: int = 999):
        self.max_words_per_chunk = max_words_per_chunk
        self.max_items_per_chunk = max_items_per_chunk
        self.max_ents_per_item = max_ents_per_item
        self.word_tokenizer = English().tokenizer
        self.meta_sections = {'see also', 'external links', 'references', 'notes', 'further reading'}
        self.max_tokens_per_item = 30

    def __call__(self, pages: Dict[DbpResource, dict], entity_labels: Dict[DbpResource, Tuple[Set[int], Set[int]]] = None) -> Dict[DbpResource, Tuple[list, list, list]]:
        if entity_labels:
            n_processes = utils.get_config('max_cpus') // 2
            # discard pages without labels
            pages = {res: page_data for res, page_data in pages.items() if entity_labels[res][0] | entity_labels[res][1]}
        else:
            n_processes = 2
            entity_labels = defaultdict(lambda: (set(), set()))

        def page_with_labels_iterator():
            for res, page_data in pages.items():
                yield res, page_data, entity_labels[res]
        page_items = tqdm(page_with_labels_iterator(), total=len(pages), desc='Tokenize Pages')
        with mp.Pool(processes=n_processes) as pool:
            return {res: tokens for res, tokens in pool.imap_unordered(self._tokenize_page, page_items, chunksize=1000) if tokens[0]}

    def _tokenize_page(self, params: Tuple[DbpResource, dict, Tuple[Set[int], Set[int]]]) -> Tuple[DbpResource, Tuple[list, list, list]]:
        """Take a resource and return list of (tokens, ws, entities) chunks."""
        res, page_data, page_labels = params
        page_tokens, page_ws, page_ents = [], [], []

        top_section_name = ''
        for section_data in page_data['sections']:
            section_name = section_data['name']
            top_section_name = section_name if section_data['level'] <= 2 else top_section_name
            if top_section_name.lower() in self.meta_sections:
                continue  # skip meta sections
            for enum_data in section_data['enums']:
                listing_tokens, listing_ws, listing_ents = self._tokenize_listing(res, top_section_name, section_name, enum_data, page_labels)
                page_tokens.extend(listing_tokens)
                page_ws.extend(listing_ws)
                page_ents.extend(listing_ents)
            for table in section_data['tables']:
                table_header, table_data = table['header'], table['data']
                listing_tokens, listing_ws, listing_ents = self._tokenize_listing(res, top_section_name, section_name, table_data, page_labels, table_header)
                page_tokens.extend(listing_tokens)
                page_ws.extend(listing_ws)
                page_ents.extend(listing_ents)
        return res, (page_tokens, page_ws, page_ents)

    def _tokenize_listing(self, res: DbpResource, top_section: str, section: str, listing_data: list, page_labels: Tuple[set, set], table_header=None) -> Tuple[list, list, list]:
        ctx_tokens, ctx_ws, ctx_ents = self._context_to_tokens([res.get_label(), top_section, section], table_header)

        max_chunk_size = self.max_words_per_chunk - len(ctx_tokens)
        listing_tokens, listing_ws, listing_ents = [], [], []
        for chunk_tokens, chunk_ws, chunk_ents in self._listing_to_token_chunks(listing_data, page_labels, max_chunk_size):
            listing_tokens.append(ctx_tokens + chunk_tokens)
            listing_ws.append(ctx_ws + chunk_ws)
            listing_ents.append(ctx_ents + chunk_ents)
        return listing_tokens, listing_ws, listing_ents

    def _context_to_tokens(self, context: List[str], table_header=None) -> Tuple[list, list, list]:
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

        ctx_ents = [WordTokenizerSpecialLabel.IGNORE.value] * len(ctx_tokens)
        return ctx_tokens, ctx_ws, ctx_ents

    def _listing_to_token_chunks(self, listing_data: list, page_labels: Tuple[set, set], max_chunk_size: int) -> Tuple[list, list, list]:
        """Converts a listing to a set of (tokens, ws, entities) chunks."""
        current_chunk_size = 0
        current_chunk_items = 0
        current_chunk_tokens, current_chunk_ws, current_chunk_ents = [], [], []
        for item in listing_data:
            # pick conversion function based on item being a list of table cells or an enum entry string
            item_to_tokens = self._row_to_tokens if isinstance(item, list) else self._entry_to_tokens
            item_tokens, item_ws, item_ents = item_to_tokens(item, page_labels)

            if not item_tokens:
                continue  # skip if no tokens are found
            item_size = len(item_tokens)
            new_chunk_size = current_chunk_size + item_size
            if not current_chunk_tokens or (new_chunk_size <= max_chunk_size and current_chunk_items < self.max_items_per_chunk):
                current_chunk_tokens.extend(item_tokens)
                current_chunk_ws.extend(item_ws)
                current_chunk_ents.extend(item_ents)
                current_chunk_size = new_chunk_size
                current_chunk_items += 1
            else:
                yield current_chunk_tokens, current_chunk_ws, current_chunk_ents
                current_chunk_tokens = item_tokens
                current_chunk_ws = item_ws
                current_chunk_ents = item_ents
                current_chunk_size = item_size
                current_chunk_items = 1
        if current_chunk_tokens:
            yield current_chunk_tokens, current_chunk_ws, current_chunk_ents

    def _entry_to_tokens(self, entry: dict, page_labels: Tuple[set, set]) -> Tuple[list, list, list]:
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
                return [], [], []
        else:  # only extract tokens and ignore entities
            entry_doc = self.word_tokenizer(entry_text)
            entry_entities = []
            valid_ents = set()

        entry_entities = [e for e in entry_entities if e['idx'] in valid_ents][:self.max_ents_per_item]
        tokens, ws, ents = self._text_to_tokens(entry_doc, entry_entities)
        tokens.insert(0, WordTokenizerSpecialToken.get_entry_by_depth(depth))
        ws.insert(0, ' ')
        ents.insert(0, WordTokenizerSpecialLabel.NO_ENTITY.value)
        return tokens, ws, ents

    def _row_to_tokens(self, row: list, page_labels: Tuple[set, set]) -> Tuple[list, list, list]:
        if page_labels[0] | page_labels[1]:  # page labels provided -> extract entities
            cell_docs = [list_nlp.parse(cell['text']) for cell in row]
            valid_ents, invalid_ents = page_labels
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
                return [], [], []
        else:  # only extract tokens and ignore entities
            cell_docs = [self.word_tokenizer(cell['text']) for cell in row]
            valid_ents = set()

        tokens, ws, ents = [], [], []
        max_ents_per_row = self.max_ents_per_item
        for cell_idx, cell in enumerate(row):
            cell_ents = [e for e in cell['entities'] if e['idx'] in valid_ents][:max_ents_per_row]
            max_ents_per_row -= len(cell_ents)
            cell_tokens, cell_ws, cell_ents = self._text_to_tokens(cell_docs[cell_idx], cell_ents)
            tokens += [WordTokenizerSpecialToken.TABLE_COL.value] + cell_tokens
            ws += [' '] + cell_ws
            ents += [WordTokenizerSpecialLabel.NO_ENTITY.value] + cell_ents
        if tokens:
            tokens[0] = WordTokenizerSpecialToken.TABLE_ROW.value  # special indicator for start of table row
        return tokens, ws, ents

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
            return tokens, ws, [WordTokenizerSpecialLabel.NO_ENTITY.value] * len(tokens)

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
        entity_pos_map = defaultdict(lambda: WordTokenizerSpecialLabel.NO_ENTITY.value)
        for ent in entities:
            start = ent['start']
            text_length = len(ent['text'])
            for i in range(start, start + text_length):
                entity_pos_map[i] = ent['idx']
        return entity_pos_map
