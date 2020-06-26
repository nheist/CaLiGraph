"""Functionality for parsing Wikipedia pages from WikiText."""

import wikitextparser as wtp
from wikitextparser import WikiText
from typing import Tuple, Optional
import re
import impl.dbpedia.util as dbp_util


def parse_page(page_markup: str) -> dict:
    """Return a single parsed page in the following hierarchical structure:

    Sections > Enums > Entries > Entities
    Sections > Tables > Rows > Columns > Entities
    """
    wiki_text = wtp.parse(page_markup)
    cleaned_wiki_text = _convert_special_enums(wiki_text)

    return {'sections': _extract_sections(cleaned_wiki_text)}


def _convert_special_enums(wiki_text: WikiText) -> WikiText:
    """Remove special templates used as enumerations from the text."""
    result = wiki_text.string
    enum_templates = [t for t in wiki_text.templates if t.name == 'columns-list']
    for et in enum_templates:
        actual_list = et.get_arg('1')
        result = result.replace(et.string, actual_list.string[1:] if actual_list else '')
    return wtp.parse(result)


def _extract_sections(wiki_text: WikiText) -> list:
    return [{
        'name': section.title.strip() if section.title.strip() else 'Main',
        'content': section.contents,
        'enums': [_extract_enum(l) for l in section.lists() if l.level == 1],
        'tables': [_extract_table(t) for t in section.tables]
    } for section in wiki_text.sections]


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
        rows = table.data()
    except IndexError:
        return []

    for row in rows:
        parsed_columns = []
        for column in row:
            plaintext, entities = _convert_markup(column)
            parsed_columns.append({
                'text': plaintext,
                'entities': entities
            })
        row_data.append(parsed_columns)
    return row_data


def _convert_markup(wiki_text: str) -> Tuple[str, list]:
    # remove all markup except wikilinks
    current = None
    new = wiki_text
    while current != new:
        current = new
        new = _remove_wikimarkup(wtp.parse(current))

    # extract wikilink-entities and generate final plain text
    entities = []
    while True:
        new, entity = _extract_entity(current)
        if entity is None:
            break
        if entity:
            entities.append(entity)
        current = new

    return current, entities


def _extract_entity(text: str) -> Tuple[str, Optional[dict]]:
    wiki_text = wtp.parse(text)
    if not wiki_text.wikilinks:
        return text, None

    link = wiki_text.wikilinks[0]
    link_text = (link.text.strip() if link.text else link.text) or link.target.strip()
    link_target = link.target[0].upper() + link.target[1:] if len(link.target) > 1 else link.target.upper()

    if '|' in link_text:
        link_text = link_text[link_text.rindex('|'):].strip()

    pre = text[:text.index(link.string)].strip()
    pre = pre + ' ' if pre else ''
    post = text[text.index(link.string)+len(link.string):].strip()
    post = ' ' + post if post else ''

    plaintext = f'{pre}{link_text}{post}'

    if '[[' in link_text:
        return plaintext, {}

    return plaintext, {
        'uri': dbp_util.name2resource(_remove_language_tag(link_target.strip())),
        'text': link_text,
        'idx': len(pre)
    }


def _remove_language_tag(link_target: str) -> str:
    if len(link_target) == 0 or link_target[0] != ':':
        return link_target
    if len(link_target) < 4 or link_target[3] != ':':
        return link_target[1:]
    return link_target[4:]


def _remove_wikimarkup(wiki_text: WikiText) -> str:
    result = wiki_text.string
    for tag in wiki_text.tags():
        if tag._match is not None:
            result = result.replace(tag.string, _wrap_in_spaces(tag.contents))
    for link in wiki_text.external_links:
        try:
            result = result.replace(link.string, _wrap_in_spaces(link.text) if link.text else ' ')
        except AttributeError:
            result = result.replace(link.string, ' ')
    for template in wiki_text.templates:
        result = result.replace(template.string, ' ')
    for comment in wiki_text.comments:
        result = result.replace(comment.string, ' ')

    return _normalize_text(result)


def _wrap_in_spaces(word: str) -> str:
    return f' {word} '


def _normalize_text(text: str) -> str:
    text = re.sub("'{2,}", '', text)
    text = re.sub(' +', ' ', text)
    return text.strip()
