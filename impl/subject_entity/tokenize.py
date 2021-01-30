"""Create word tokens (and labels) from parsed wiki markup."""

from spacy.lang.en import English
from collections import defaultdict
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.dbpedia.heuristics as dbp_heur
import impl.category.store as cat_store
import impl.category.base as cat_base
import impl.list.mapping as list_mapping
import impl.list.nlp as list_nlp
import impl.list.util as list_util
import wikitextparser as wtp


# size of token batch for BERT
MAX_EXAMPLE_SIZE = 512 / 2  # use a lower example size to account for WordPiece Tokenization
# special tokens
TOKEN_CTX = '[CTX]'
TOKEN_SEP = '[SEP]'
TOKEN_ROW = '[ROW]'
TOKEN_COL = '[COL]'
TOKENS_ENTRY = [f'E{i}' for i in range(1, 6)]
ADDITIONAL_SPECIAL_TOKENS = TOKENS_ENTRY + [TOKEN_ROW, TOKEN_COL]


# EXTRACTION OF TOKENS


def page_to_tokens(params) -> tuple:
    """Take a page and return list of token-lists."""
    page_uri, page_data = params
    is_list_page = list_util.is_listpage(page_uri)
    page_name = list_util.listpage2name(page_uri) if is_list_page else dbp_util.resource2name(page_uri)
    page_tokens = []

    top_section_name = ''
    for section_data in page_data['sections']:
        section_name = section_data['name']
        top_section_name = section_name if section_data['level'] <= 2 else top_section_name
        for enum_data in section_data['enums']:
            context_tokens = _context_to_tokens([page_name, top_section_name, section_name])
            max_group_size = MAX_EXAMPLE_SIZE - len(context_tokens)
            for group_tokens in _listing_to_token_groups(enum_data, max_group_size):
                page_tokens.append(context_tokens + group_tokens)
        for table in section_data['tables']:
            table_header, table_data = table['header'], table['data']
            context_tokens = _context_to_tokens([page_name, top_section_name, section_name], table_header)
            max_group_size = MAX_EXAMPLE_SIZE - len(context_tokens)
            for group_tokens in _listing_to_token_groups(table_data, max_group_size):
                page_tokens.append(context_tokens + group_tokens)
    return page_uri, page_tokens


def _listing_to_token_groups(listing_data, max_group_size):
    current_group_size = 0
    current_group_tokens = []
    for item in listing_data:
        # pick conversion function based on item being a list of table cells or an enum entry string
        item_to_tokens = _row_to_tokens if type(item) == list else _entry_to_tokens

        item_tokens = item_to_tokens(item)
        if not item_tokens:
            continue  # skip if no tokens are found
        item_size = len(item_tokens)
        new_group_size = current_group_size + item_size
        if not current_group_tokens or new_group_size <= max_group_size:
            current_group_tokens.extend(item_tokens)
            current_group_size = new_group_size
        else:
            yield current_group_tokens
            current_group_tokens = []
            current_group_size = 0
    if current_group_tokens:
        yield current_group_tokens


def _entry_to_tokens(entry) -> list:
    entry_doc = list_nlp.parse(entry['text'])
    depth = entry['depth']
    tokens, _ = _text_to_tokens(entry_doc, None, None)
    return [f'[E{depth}]'] + tokens


def _row_to_tokens(row) -> list:
    tokens = []
    for cell in row:
        cell_tokens, _ = _text_to_tokens(list_nlp.parse(cell['text']), None, None)
        tokens += [TOKEN_COL] + cell_tokens
    if tokens:
        tokens[0] = TOKEN_ROW  # make start of table row similar to start of enum entry
    return tokens


# EXTRACTION OF TOKENS WITH LABELS


LABEL_NONE = 'O'
LABEL_BEGIN = 'B-SE'
LABEL_IN = 'I-SE'
ALL_LABELS = [LABEL_NONE, LABEL_BEGIN, LABEL_IN]


def page_to_tokens_and_labels(params) -> tuple:
    """Take a page and return list of (token,label)-lists. Labels for a list of tokens may be None."""
    page_uri, page_data, graph = params
    page_name = list_util.listpage2name(page_uri)
    top_section_name = ''
    page_tokens, page_labels = [], []
    positive_SEs, negative_SEs = _compute_labeled_entities_for_listpage(page_uri, page_data, graph)
    if not positive_SEs and not negative_SEs:
        return page_tokens, page_labels

    for section_data in page_data['sections']:
        section_name = section_data['name']
        top_section_name = section_name if section_data['level'] <= 2 else top_section_name
        for enum_data in section_data['enums']:
            context_tokens = _context_to_tokens([page_name, top_section_name, section_name])
            max_group_size = MAX_EXAMPLE_SIZE - len(context_tokens)
            for group_tokens, group_labels in _listing_to_token_label_groups(enum_data, positive_SEs, negative_SEs, max_group_size):
                page_tokens.append(context_tokens + group_tokens)
                context_labels = [LABEL_NONE] * len(context_tokens)
                page_labels.append(context_labels + group_labels)
        for table in section_data['tables']:
            table_header, table_data = table['header'], table['data']
            context_tokens = _context_to_tokens([page_name, top_section_name, section_name], table_header)
            max_group_size = MAX_EXAMPLE_SIZE - len(context_tokens)
            for group_tokens, group_labels in _listing_to_token_label_groups(table_data, positive_SEs, negative_SEs, max_group_size):
                page_tokens.append(context_tokens + group_tokens)
                context_labels = [LABEL_NONE] * len(context_tokens)
                page_labels.append(context_labels + group_labels)
    return page_tokens, page_labels


def _listing_to_token_label_groups(listing_data, positive_SEs, negative_SEs, max_group_size):
    current_group_size = 0
    current_group_tokens = []
    current_group_labels = []
    for item in listing_data:
        # pick conversion function based on item being a list of table cells or an enum entry string
        item_to_tokens = _row_to_tokens_and_labels if type(item) == list else _entry_to_tokens_and_labels

        item_tokens, item_labels = item_to_tokens(item, positive_SEs, negative_SEs)
        if not item_tokens:
            continue  # skip if no labeled entities are found
        item_size = len(item_tokens)
        new_group_size = current_group_size + item_size
        if not current_group_tokens or new_group_size <= max_group_size:
            current_group_tokens.extend(item_tokens)
            current_group_labels.extend(item_labels)
            current_group_size = new_group_size
        else:
            yield current_group_tokens, current_group_labels
            current_group_tokens = []
            current_group_labels = []
            current_group_size = 0
    if current_group_tokens:
        yield current_group_tokens, current_group_labels


def _entry_to_tokens_and_labels(entry, positive_SEs, negative_SEs) -> tuple:
    entry_text = entry['text']
    entry_doc = list_nlp.parse(entry_text)
    depth = entry['depth']

    # check whether entry is valid training data
    entities = list(entry['entities'])
    entities.extend(_get_untagged_entities(entry_doc, entities))
    entity_names = {e['name'] for e in entities}
    if all(en not in positive_SEs for en in entity_names) and any(en not in negative_SEs for en in entity_names):
        return [], []

    # extract tokens and labels
    tokens, labels = _text_to_tokens(entry_doc, entities, positive_SEs)
    if not tokens or not labels:
        return [], []
    return [f'[E{depth}]'] + tokens, [LABEL_NONE] + labels


def _row_to_tokens_and_labels(row, positive_SEs, negative_SEs) -> tuple:
    cell_docs = [list_nlp.parse(cell['text']) for cell in row]
    # check whether row is valid training data
    row_entities = []
    for cell_idx, cell in enumerate(row):
        cell_doc = cell_docs[cell_idx]
        cell_entities = list(cell['entities'])
        row_entities.extend(cell_entities)
        row_entities.extend(_get_untagged_entities(cell_doc, cell_entities))
    entity_names = {e['name'] for e in row_entities}
    if all(en not in positive_SEs for en in entity_names) and any(en not in negative_SEs for en in entity_names):
        return [], []

    # extract tokens and labels
    tokens, labels = [], []
    for cell_idx, cell in enumerate(row):
        cell_doc = cell_docs[cell_idx]
        cell_entities = list(cell['entities'])
        cell_tokens, cell_labels = _text_to_tokens(cell_doc, cell_entities, positive_SEs)
        tokens += [TOKEN_COL] + cell_tokens
        labels += [LABEL_NONE] + cell_labels
    if tokens:
        tokens[0] = TOKEN_ROW  # make start of table row more specific
    return tokens, labels


def _get_untagged_entities(doc, entities: list):
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
        while (end-1) in entity_character_idxs and start < end:
            end -= 1
            text = text[:-1]
        if not entity_character_idxs.intersection(set(range(start, end))) and len(text.strip()) > 1:
            untagged_entities.append({'name': 'DEFINITELY-NOT-IN-EXISTING-ENTITIES'})
    return untagged_entities


def _create_entity_label_map(entities: list, subject_entities: set):
    label_map = defaultdict(lambda: LABEL_NONE)
    for ent in entities:
        if ent['name'] not in subject_entities:
            continue
        idx = ent['idx']
        text_length = len(ent['text'])
        for i in range(idx, idx + text_length):
            label = LABEL_BEGIN if i == idx else LABEL_IN
            label_map[i] = label
    return label_map


def _compute_labeled_entities_for_listpage(page_uri, page_data, graph) -> tuple:
    positive_SEs, negative_SEs = set(), set()
    # compute potential subject entities for list page
    page_potential_SEs = {dbp_util.resource2name(res) for cat in _get_category_descendants_for_list(page_uri) for res in cat_store.get_resources(cat)}
    # compute types of list page
    page_types = {t for n in graph.get_nodes_for_part(page_uri) for t in dbp_store.get_independent_types(graph.get_transitive_dbpedia_types(n))}
    page_disjoint_types = {dt for t in page_types for dt in dbp_heur.get_disjoint_types(t)}
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
    cat_graph = cat_base.get_merged_graph()
    mapped_categories = {x for cat in _get_categories_for_list(listpage_uri) for x in cat_graph.get_nodes_for_category(cat)}
    descendant_categories = {descendant for cat in mapped_categories for descendant in cat_graph.descendants(cat)}
    for cat in mapped_categories | descendant_categories:
        categories.update(cat_graph.get_categories(cat))
    return categories


def _get_categories_for_list(listpage_uri: str) -> set:
    """Return category that is mapped to the list page."""
    return list_mapping.get_equivalent_categories(listpage_uri) | list_mapping.get_parent_categories(listpage_uri)


# SHARED METHODS


def _context_to_tokens(ctx: list, table_header=[]) -> list:
    ctx_tokens = []

    # add basic context, separated by [CTX] special tokens
    spacy_tokenizer = _get_spacy_tokenizer()
    for text in ctx:
        plain_text = wtp.parse(text).plain_text()
        ctx_tokens.extend([w.text for w in spacy_tokenizer(plain_text)])
        ctx_tokens.append(TOKEN_CTX)

    # add table header if available
    for cell in table_header:
        ctx_tokens.extend([w.text for w in spacy_tokenizer(cell['text'])])
        ctx_tokens.append(TOKEN_COL)

    ctx_tokens[-1] = TOKEN_SEP  # replace last token with final context separator
    return ctx_tokens


def _get_spacy_tokenizer():
    global __SPACY_TOKENIZER__
    if '__SPACY_TOKENIZER__' not in globals():
        nlp = English()
        __SPACY_TOKENIZER__ = nlp.Defaults.create_tokenizer(nlp)
    return __SPACY_TOKENIZER__


def _text_to_tokens(doc, entities: list, subject_entities: set) -> tuple:
    if not entities or not subject_entities:
        tokens = [w.text for w in doc]
        return tokens, [LABEL_NONE] * len(tokens)

    tokens, labels = [], []
    entity_label_map = _create_entity_label_map(entities, subject_entities)
    current_position = 0
    for w in doc:
        tokens.append(w.text)
        labels.append(entity_label_map[current_position])
        current_position += len(w.text_with_ws)
    return tokens, labels
