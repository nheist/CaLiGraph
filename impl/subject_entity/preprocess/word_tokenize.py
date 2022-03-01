from typing import Tuple, List, Optional
from spacy.lang.en import English
from collections import defaultdict
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.dbpedia.heuristics as dbp_heur
import impl.category.store as cat_store
from impl import category
import impl.listpage.mapping as list_mapping
import impl.util.spacy.listing_parser as list_nlp
import impl.listpage.util as list_util
import impl.wikipedia.wikimarkup_parser as wmp
from tqdm import tqdm
from enum import Enum


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
        return {cls.TABLE_ROW, cls.ENTRY_L1, cls.ENTRY_L2, cls.ENTRY_L3}

    @classmethod
    def get_entry_by_depth(cls, depth: int):
        if depth == 1:
            return cls.ENTRY_L1
        elif depth == 2:
            return cls.ENTRY_L2
        elif depth >= 3:
            return cls.ENTRY_L3
        raise ValueError(f'Trying to retrieve a BERT special token for an entry of depth {depth}.')


class WordTokenizer:
    """Takes parsed wiki markup and splits it into word tokens while preserving entity labels."""
    def __init__(self, max_words_per_chunk=384):
        self.word_tokenizer = English().tokenizer
        self.max_words_per_chunk = max_words_per_chunk
        self.meta_sections = {'see also', 'external links', 'references', 'notes'}

    def __call__(self, pages: dict, graph=None) -> dict:
        if graph:
            return dict([self._tokenize_with_entities((page_uri, page_data, graph))
                         for page_uri, page_data in tqdm(pages.items(), desc='Tokenize Pages (train)')])
        else:
            return dict([self._tokenize(page_uri_with_data)
                         for page_uri_with_data in tqdm(pages.items(), desc='Tokenize Pages (all)')])

    def _tokenize_with_entities(self, params) -> Tuple[str, Tuple[list, list]]:
        """Take a page and return list of (tokens, entities) chunks. If not existing, entity for a token is None."""
        page_uri, page_data, graph = params
        positive_SEs, negative_SEs = _compute_labeled_entities_for_listpage(page_uri, page_data, graph)
        page_name = list_util.listpage2name(page_uri)

        page_tokens, page_ents = [], []
        if not positive_SEs | negative_SEs:
            # can't produce valid entity labels without positive/negative SEs, so we discard the page
            return page_uri, (page_tokens, page_ents)

        top_section_name = ''
        for section_data in page_data['sections']:
            section_name = section_data['name']
            top_section_name = section_name if section_data['level'] <= 2 else top_section_name
            if top_section_name.lower() in self.meta_sections:
                continue  # skip meta sections

            for enum_data in section_data['enums']:
                context_tokens, _ = self._context_to_tokens([page_name, top_section_name, section_name])
                context_ents = [None] * len(context_tokens)

                max_chunk_size = self.max_words_per_chunk - len(context_tokens)
                for chunk_tokens, chunk_ents in self._listing_to_token_entity_chunks(enum_data, positive_SEs, negative_SEs, max_chunk_size):
                    page_tokens.append(context_tokens + chunk_tokens)
                    page_ents.append(context_ents + chunk_ents)

            for table in section_data['tables']:
                table_header, table_data = table['header'], table['data']
                context_tokens, _ = self._context_to_tokens([page_name, top_section_name, section_name], table_header)
                context_ents = [None] * len(context_tokens)

                max_chunk_size = self.max_words_per_chunk - len(context_tokens)
                for chunk_tokens, chunk_ents in self._listing_to_token_entity_chunks(table_data, positive_SEs, negative_SEs, max_chunk_size):
                    page_tokens.append(context_tokens + chunk_tokens)
                    page_ents.append(context_ents + chunk_ents)
        return page_uri, (page_tokens, page_ents)

    def _context_to_tokens(self, context: List[str], table_header=None) -> Tuple[list, list]:
        """Converts list of context strings (and table header) to context tokens."""
        ctx_tokens, ctx_ws = [], []

        # add listing context, separated by special context tokens
        for text in context:
            doc = self.word_tokenizer(wmp.wikitext_to_plaintext(text))
            ctx_tokens.extend([w.text for w in doc] + [BertSpecialToken.CONTEXT_SEP])
            ctx_ws.extend([w.whitespace_ for w in doc] + [' '])

        # add table header if available
        if table_header:
            for cell in table_header:
                doc = self.word_tokenizer(cell['text'])
                ctx_tokens.extend([w.text for w in doc] + [BertSpecialToken.TABLE_COL])
                ctx_ws.extend([w.whitespace_ for w in doc] + [' '])

        ctx_tokens[-1] = BertSpecialToken.CONTEXT_END  # replace last token with final context separator
        return ctx_tokens, ctx_ws

    def _listing_to_token_entity_chunks(self, listing_data: list, positive_SEs: set, negative_SEs: set, max_group_size: int):
        """Converts a listing to a set of (tokens, entities) chunks."""
        current_group_size = 0
        current_group_tokens = []
        current_group_ents = []
        for item in listing_data:
            # pick conversion function based on item being a list of table cells or an enum entry string
            item_to_tokens = self._row_to_tokens_and_entities if type(item) == list else self._entry_to_tokens_and_entities
            item_tokens, item_ents = item_to_tokens(item, positive_SEs, negative_SEs)

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

    def _entry_to_tokens_and_entities(self, entry: dict, positive_SEs: set, negative_SEs: set) -> Tuple[list, list]:
        entry_text = entry['text']
        entry_doc = list_nlp.parse(entry_text)
        depth = entry['depth']

        # check whether entry is valid training data
        entities = list(entry['entities'])
        entities.extend(self._find_untagged_entities(entry_doc, entities))
        entity_names = {e['name'] for e in entities}
        if all(en not in positive_SEs for en in entity_names) and any(en not in negative_SEs for en in entity_names):
            # discard entry if 1) there are no subject entities we know of and
            # 2) we are unsure about other entities (i.e., they MAY be subject entities)
            return [], []

        # extract tokens and entities
        tokens, _, ents = self._text_to_tokens(entry_doc, entities, positive_SEs)
        if not tokens or not ents:
            return [], []
        return [BertSpecialToken.get_entry_by_depth(depth)] + tokens, [None] + ents

    def _row_to_tokens_and_entities(self, row: list, positive_SEs: set, negative_SEs: set) -> Tuple[list, list]:
        cell_docs = [list_nlp.parse(cell['text']) for cell in row]
        # check whether row is valid training data
        row_entities = []
        for cell_idx, cell in enumerate(row):
            cell_doc = cell_docs[cell_idx]
            cell_entities = list(cell['entities'])
            row_entities.extend(cell_entities)
            row_entities.extend(self._find_untagged_entities(cell_doc, cell_entities))
        entity_names = {e['name'] for e in row_entities}
        if all(en not in positive_SEs for en in entity_names) and any(en not in negative_SEs for en in entity_names):
            # discard entry if 1) there are no subject entities we know of and
            # 2) we are unsure about other entities (i.e., they MAY be subject entities)
            return [], []

        # extract tokens and entities
        tokens, ents = [], []
        for cell_idx, cell in enumerate(row):
            cell_doc = cell_docs[cell_idx]
            cell_entities = list(cell['entities'])
            cell_tokens, _, cell_ents = self._text_to_tokens(cell_doc, cell_entities, positive_SEs)
            tokens += [BertSpecialToken.TABLE_COL] + cell_tokens
            ents += [None] + cell_ents
        if tokens:
            tokens[0] = BertSpecialToken.TABLE_ROW  # special indicator for start of table row
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
            tokens += [BertSpecialToken.TABLE_COL] + cell_tokens
            ws += [' '] + cell_ws
        if tokens:
            tokens[0] = BertSpecialToken.TABLE_ROW
        return tokens, ws

    def _text_to_tokens(self, doc, entities: Optional[list], subject_entities: Optional[set]) -> Tuple[list, list, list]:
        """Transforms a spacy doc (and entity info) to lists of tokens, whitespaces(, and entities)."""
        if not entities or not subject_entities:
            tokens = [w.text for w in doc]
            ws = [w.whitespace_ for w in doc]
            return tokens, ws, [None] * len(tokens)

        tokens, ws, token_ents = [], [], []
        entity_pos_map = self._create_entity_position_map(entities, subject_entities)
        current_position = 0
        for w in doc:
            tokens.append(w.text)
            ws.append(w.whitespace_)
            token_ents.append(entity_pos_map[current_position])
            current_position += len(w.text_with_ws)
        return tokens, ws, token_ents

    def _create_entity_position_map(self, entities: list, subject_entities: set):
        """Index subject entities by their text position."""
        entity_pos_map = defaultdict(lambda: None)
        valid_ents = [e for e in entities if e['name'] in subject_entities]
        for ent in valid_ents:
            idx = ent['idx']
            text_length = len(ent['text'])
            for i in range(idx, idx + text_length):
                entity_pos_map[i] = ent['name']
        return entity_pos_map


def _compute_labeled_entities_for_listpage(page_uri: str, page_data: dict, graph) -> tuple:
    """Retrieve all entities of a list page that are (based on a heuristic) subject entities or non-subject entities."""
    positive_SEs, negative_SEs = set(), set()
    # compute potential subject entities for list page
    page_potential_SEs = {dbp_util.resource2name(res) for cat in _get_category_descendants_for_list(page_uri) for res in cat_store.get_resources(cat)}
    # compute types of list page
    page_types = {t for n in graph.get_nodes_for_part(page_uri) for t in dbp_store.get_independent_types(graph.get_transitive_dbpedia_type_closure(n))}
    page_disjoint_types = {dt for t in page_types for dt in dbp_heur.get_all_disjoint_types(t)}
    # collect all linked entities on the page
    page_entities = {ent['name'] for s in page_data['sections'] for enum in s['enums'] for entry in enum for ent in entry['entities']}
    page_entities.update({ent['name'] for s in page_data['sections'] for table in s['tables'] for row in table['data'] for cell in row for ent in cell['entities']})
    for ent in page_entities:
        ent_uri = dbp_util.name2resource(ent)
        if not dbp_store.is_possible_resource(ent_uri):
            negative_SEs.add(ent)
        elif ent in page_potential_SEs:
            positive_SEs.add(ent)
        elif page_disjoint_types.intersection(dbp_store.get_types(ent_uri)):
            negative_SEs.add(ent)
    return positive_SEs, negative_SEs


def _get_category_descendants_for_list(listpage_uri: str) -> set:
    """Return the category that is most closely related to the given list page as well as all of its children."""
    categories = set()
    cat_graph = category.get_merged_graph()
    mapped_categories = {x for cat in _get_categories_for_list(listpage_uri) for x in cat_graph.get_nodes_for_category(cat)}
    descendant_categories = {descendant for cat in mapped_categories for descendant in cat_graph.descendants(cat)}
    for cat in mapped_categories | descendant_categories:
        categories.update(cat_graph.get_categories(cat))
    return categories


def _get_categories_for_list(listpage_uri: str) -> set:
    """Return category that is mapped to the list page."""
    return list_mapping.get_equivalent_categories(listpage_uri) | list_mapping.get_parent_categories(listpage_uri)
