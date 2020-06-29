"""Functionality for parsing Wikipedia pages from WikiText."""

import wikitextparser as wtp
from wikitextparser import WikiText
from typing import Tuple
import impl.dbpedia.util as dbp_util
import util
import re


def parse_page(page_markup: str) -> dict:
    """Return a single parsed page in the following hierarchical structure:

    Sections > Enums > Entries > Entities
    Sections > Tables > Rows > Columns > Entities
    """
    wiki_text = wtp.parse(page_markup)
    if not _is_page_useful(wiki_text):
        return None

    cleaned_wiki_text = _prepare_wikitext(wiki_text)
    if not _is_page_useful(cleaned_wiki_text):
        return None

    return {'sections': _extract_sections(cleaned_wiki_text)}


def _is_page_useful(wiki_text: WikiText) -> bool:
    return len(wiki_text.get_lists()) + len(wiki_text.get_tables()) > 0


def _prepare_wikitext(wiki_text: WikiText) -> WikiText:
    """Convert special templates used as enumerations from the text and remove bolds&italics."""
    result = wiki_text.string
    # convert enumeration templates
    enum_templates = [t for t in wiki_text.templates if t.name == 'columns-list']
    for et in enum_templates:
        actual_list = et.get_arg('1')
        result = result.replace(et.string, actual_list.string[1:] if actual_list else '')
    # remove bolds and italics
    #result = re.sub(r"'{2,}", "", result)
    # convert html whitespaces
    result = result.replace('&nbsp;', ' ')
    return wtp.parse(result)


def _extract_sections(wiki_text: WikiText) -> list:
    return [{
        'index': section_idx,
        'name': section.title.strip() if section.title and section.title.strip() else 'Main',
        'markup': section.contents,
        'text': _wikitext_to_plaintext(section),
        'enums': [_extract_enum(l) for l in section.get_lists()],
        'tables': [_extract_table(t) for t in section.get_tables()]
    } for section_idx, section in enumerate(wiki_text.sections)]


def _extract_enum(l: wtp.WikiList) -> list:
    entries = []
    for item_idx, item_text in enumerate(l.items):
        plaintext, entities = _convert_markup(item_text)
        sublists = l.sublists(item_idx)
        entries.append({
            'text': plaintext,
            'depth': l.level,
            'leaf': len(sublists) == 0,
            'entities': entities
        })
        for sl in sublists:
            entries.extend(_extract_enum(sl))
    return entries


def _extract_table(table: wtp.Table) -> list:
    row_data = []
    try:
        rows = table.data(strip=True)
    except:
        return []

    for row in rows:
        parsed_cells = []
        for cell in row:
            plaintext, entities = _convert_markup(str(cell))
            parsed_cells.append({
                'text': plaintext,
                'entities': entities
            })
        row_data.append(parsed_cells)
    return row_data


def _convert_markup(wiki_text: str) -> Tuple[str, list]:
    parsed_text = wtp.parse(wiki_text)
    plain_text = _wikitext_to_plaintext(parsed_text).strip()

    # extract wikilink-entities with correct positions in plain text
    entities = []
    current_entity_index = 0
    for w in parsed_text.wikilinks:
        # retrieve entity text
        text = (w.text or w.target).strip()
        if '|' in text:  # hot-fixing broken markup
            util.get_logger().debug(f'Found broken wikilink with text "{text}" in {w}')
            text = text[text.rindex('|'):].strip()
        if not text:
            continue  # skip entity with empty text

        # retrieve entity position
        if text not in plain_text[current_entity_index:]:
            continue  # skip entity with a text that can not be located
        entity_position = current_entity_index + plain_text[current_entity_index:].index(text)
        current_entity_index = entity_position + len(text)
        entities.append({'idx': entity_position, 'text': text, 'uri': _convert_target_to_uri(w.target)})
    return plain_text, entities


def _wikitext_to_plaintext(parsed_text: wtp.WikiText) -> str:
    for t in parsed_text.get_tags():
        if not t._match:
            t[:] = ''  # manually remove tags without _match as they cause errors in the parser
    return parsed_text.plain_text()


def _convert_target_to_uri(link_target: str) -> str:
    link_target = _remove_language_tag(link_target.strip())
    link_target = link_target[0].upper() + link_target[1:]
    return dbp_util.name2resource(link_target)


def _remove_language_tag(link_target: str) -> str:
    if len(link_target) == 0 or link_target[0] != ':':
        return link_target
    if len(link_target) < 4 or link_target[3] != ':':
        return link_target[1:]
    return link_target[4:]
