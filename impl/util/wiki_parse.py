"""Functionality for parsing Wikipedia pages from WikiText."""

import wikitextparser as wtp
from wikitextparser import WikiText
from typing import Tuple, Optional
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import re


LISTING_INDICATORS = ('*', '#', '{|')
VALID_ENUM_PATTERNS = (r'\#', r'\*')
PAGE_TYPE_ENUM, PAGE_TYPE_TABLE = 'enum', 'table'


def parse_page(page_markup: str) -> Optional[dict]:
    """Return a single parsed page in the following hierarchical structure:

    Sections > Enums > Entries > Entities
    Sections > Tables > Rows > Columns > Entities
    """
    if not any(indicator in page_markup for indicator in LISTING_INDICATORS):
        return None  # early return of 'None' if page contains no listings at all

    # prepare markup for parsing
    page_markup = page_markup.replace('&nbsp;', ' ')  # replace html whitespaces
    page_markup = re.sub(r"'{2,}", '', page_markup)  # remove bold and italic markers

    wiki_text = wtp.parse(page_markup)
    if not _is_page_useful(wiki_text):
        return None

    cleaned_wiki_text = _convert_special_enums(wiki_text)
    cleaned_wiki_text = _remove_enums_within_tables(cleaned_wiki_text)
    if not _is_page_useful(cleaned_wiki_text):
        return None

    # collect all wikilink occurrences and their string pattern for a later expansion
    text_to_wikilink = {wl.text or wl.target: wl.string for wl in wiki_text.wikilinks if not (wl.target.startswith('File:') or wl.target.startswith('Image:'))}
    pattern_to_wikilink = {r'(?<![|\[])\b' + re.escape(text) + r'\b(?![|\]])': wl for text, wl in text_to_wikilink.items()}

    sections = _extract_sections(cleaned_wiki_text, pattern_to_wikilink)
    types = set()
    if any(len(s['enums']) > 0 for s in sections):
        types.add(PAGE_TYPE_ENUM)
    if any(len(s['tables']) > 0 for s in sections):
        types.add(PAGE_TYPE_TABLE)
    return {'sections': sections, 'types': types}


def _is_page_useful(wiki_text: WikiText) -> bool:
    # ignore pages without any lists and pages with very small lists (e.g. redirect pages have a list with length of 1)
    return len(wiki_text.get_lists(VALID_ENUM_PATTERNS)) + len(wiki_text.get_tables()) >= 3


def _convert_special_enums(wiki_text: WikiText) -> WikiText:
    """Convert special templates used as enumerations from the text."""
    # convert enumeration templates
    enum_templates = [t for t in wiki_text.templates if t.name == 'columns-list']
    if enum_templates:
        result = wiki_text.string
        for et in enum_templates:
            actual_list = et.get_arg('1')
            result = result.replace(et.string, actual_list.string[1:] if actual_list else '')
        return wtp.parse(result)
    return wiki_text


def _remove_enums_within_tables(wiki_text: WikiText) -> WikiText:
    """Remove any enumeration markup that is contained within a table."""
    something_changed = False
    for t in wiki_text.tables:
        for row in t.cells():
            for cell in row:
                if cell:
                    for lst in cell.get_lists(VALID_ENUM_PATTERNS):
                        lst.convert('')
                        something_changed = True
    return wtp.parse(wiki_text.string) if something_changed else wiki_text


def _extract_sections(wiki_text: WikiText, pattern_to_wikilink: dict) -> list:
    sections = []
    for section_idx, section in enumerate(wiki_text.get_sections(include_subsections=False)):
        #markup_without_lists = _remove_listing_markup(section)
        #text, entities = _convert_markup(markup_without_lists)
        enums = [_extract_enum(l, pattern_to_wikilink) for l in section.get_lists(VALID_ENUM_PATTERNS)]
        tables = [_extract_table(t, pattern_to_wikilink) for t in section.get_tables()]
        sections.append({
            'index': section_idx,
            'name': section.title.strip() if section.title and section.title.strip() else 'Main',
            'level': section.level,
            #'text': text,
            #'entities': entities,
            'enums': [e for e in enums if len(e) > 2],
            'tables': [t for t in tables if t]
        })
    return sections


def _remove_listing_markup(wiki_text: WikiText) -> str:
    result = wiki_text.string
    for indicator in LISTING_INDICATORS:
        if indicator in result:
            result = result[:result.index(indicator)]
    return result


def _extract_enum(l: wtp.WikiList, pattern_to_wikilink: dict) -> list:
    entries = []
    for item_idx, item_text in enumerate(l.items):
        plaintext, entities = _convert_markup(item_text, pattern_to_wikilink)
        sublists = l.sublists(item_idx)
        entries.append({
            'text': plaintext,
            'depth': l.level,
            'leaf': len(sublists) == 0,
            'entities': entities
        })
        for sl in sublists:
            entries.extend(_extract_enum(sl, pattern_to_wikilink))
    return entries


def _extract_table(table: wtp.Table, pattern_to_wikilink: dict) -> Optional[dict]:
    row_header = []
    row_data = []
    try:
        rows = table.data(strip=True, span=True)
        rows_with_spans = table.data(strip=True, span=False)
    except:
        return None
    for row_idx, row in enumerate(rows):
        if len(row) < 2 or len(row) > 100:
            # ignore tables with only one or more than 100 columns (likely irrelevant or markup error)
            return None
        parsed_cells = []
        for cell in row:
            plaintext, entities = _convert_markup(str(cell), pattern_to_wikilink)
            parsed_cells.append({
                'text': plaintext,
                'entities': entities
            })
        if _is_header_row(table, row_idx):
            row_header = parsed_cells
        else:
            if len(rows_with_spans) > row_idx and len(row) == len(rows_with_spans[row_idx]):
                # only use rows that are not influenced by row-/colspan
                row_data.append(parsed_cells)
    if len(row_data) < 2:
        return None  # ignore tables with less than 2 data rows
    return {'header': row_header, 'data': row_data}


def _is_header_row(table: wtp.Table, row_idx: int) -> bool:
    if row_idx == 0:
        return True
    cells = table.cells(row=row_idx, span=True)
    is_header_cell = lambda cell: str(cell).strip().startswith('!')
    return is_header_cell(cells[0]) & is_header_cell(cells[1])


def _convert_markup(wiki_text: str, pattern_to_wikilink: dict) -> Tuple[str, list]:
    parsed_text = wtp.parse(_expand_wikilinks(wiki_text, pattern_to_wikilink))
    plain_text = _wikitext_to_plaintext(parsed_text).strip()

    # extract wikilink-entities with correct positions in plain text
    entities = []
    current_entity_index = 0
    for w in parsed_text.wikilinks:
        # retrieve entity text
        text = (w.text or w.target).strip()
        if w.target.startswith('File:') or w.target.startswith('Image:'):
            continue  # ignore files and images
        if '|' in text:  # deal with invalid markup in wikilinks
            text = text[text.rindex('|')+1:].strip()
        if not text:
            continue  # skip entity with empty text

        # retrieve entity position
        if text not in plain_text[current_entity_index:]:
            continue  # skip entity with a text that can not be located
        entity_position = current_entity_index + plain_text[current_entity_index:].index(text)
        current_entity_index = entity_position + len(text)
        entity_uri = _convert_target_to_uri(w.target)
        if entity_uri:
            entities.append({'idx': entity_position, 'text': text, 'uri': entity_uri})
    return plain_text, entities


def _expand_wikilinks(wiki_text: str, pattern_dict: dict) -> str:
    """Replace all textual occurrences of an entity with the respective wikilink."""
    for pattern, wl in pattern_dict.items():
        try:
            wiki_text = re.sub(pattern, wl, wiki_text)
        except:
            pass  # ignore any regex key errors
    return wiki_text


def _wikitext_to_plaintext(parsed_text: wtp.WikiText) -> str:
    # bolds and italics are already removed during preprocessing to reduce runtime
    result = parsed_text.plain_text(replace_bolds_and_italics=False).strip(" '\t\n")
    result = re.sub(r'\n+', '\n', result)
    result = re.sub(r' +', ' ', result)
    return result


def _convert_target_to_uri(link_target: str) -> Optional[str]:
    link_target = _remove_language_tag(link_target.strip())
    if not link_target:
        return None
    resource_uri = dbp_util.name2resource(link_target[0].upper() + link_target[1:])
    redirected_uri = dbp_store.resolve_redirect(resource_uri)
    if dbp_store.is_possible_resource(redirected_uri) and '#' not in redirected_uri:
        # return redirected uri only if it is an own Wikipedia article and it does not point to an article section
        return redirected_uri
    else:
        return resource_uri


def _remove_language_tag(link_target: str) -> str:
    if len(link_target) == 0 or link_target[0] != ':':
        return link_target
    if len(link_target) < 4 or link_target[3] != ':':
        return link_target[1:]
    return link_target[4:]
