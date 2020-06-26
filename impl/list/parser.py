"""Functionality for parsing Wikipedia list pages from WikiText."""

import wikitextparser as wtp
from wikitextparser import WikiText
from regex._regex_core import error as RegexError
import util
import impl.list.store as list_store
import impl.util.wiki_parse as wiki_parse


# TODO: replace functionality with util.wiki_parse
LIST_TYPE_ENUM, LIST_TYPE_TABLE, LIST_TYPE_NONE = 'list_type_enum', 'list_type_table', 'list_type_none'


def get_parsed_listpages() -> dict:
    """Return parsed list pages as structured dictionary."""
    global __PARSED_LISTPAGES__
    if '__PARSED_LISTPAGES__' not in globals():
        __PARSED_LISTPAGES__ = util.load_or_create_cache('dbpedia_listpage_parsed', _compute_parsed_listpages)

    return __PARSED_LISTPAGES__


def _compute_parsed_listpages() -> dict:
    return {lp: _parse_listpage(lp, list_store.get_listpage_markup(lp)) for lp in list_store.get_listpages()}


def _parse_listpage(listpage_uri: str, listpage_markup: str) -> dict:
    """Return a single parsed list page in the following hierarchical structure:

    Sections > Entries > Entities (for enumeration list pages)
    Sections > Tables > Rows > Columns > Entities (for table list pages)
    """
    wiki_text = wtp.parse(listpage_markup)
    cleaned_wiki_text = wiki_parse._convert_special_enums(wiki_text)

    list_type = _get_list_type(cleaned_wiki_text)
    result = {
        'uri': listpage_uri,
        'type': list_type
    }

    if list_type != LIST_TYPE_NONE:
        result['sections'] = _extract_sections(list_type, cleaned_wiki_text)

    return result


# TODO: ignore enumerations that are contained in a table
def _get_list_type(wiki_text: WikiText) -> str:
    """Return layout type of the list page by counting whether we have more table rows or enumeration entries."""
    try:
        enum_entry_count = len([entry for enum in wiki_text.lists(pattern=r'\*+') for entry in enum.items])
        table_row_count = len([row for table in wiki_text.tables for row in table.data(span=False)])

        if not enum_entry_count and not table_row_count:
            return LIST_TYPE_NONE
        elif enum_entry_count > table_row_count:
            return LIST_TYPE_ENUM
        else:
            return LIST_TYPE_TABLE
    except RegexError:
        return LIST_TYPE_NONE


def _extract_sections(list_type: str, wiki_text: WikiText) -> list:
    result = []

    if list_type == LIST_TYPE_ENUM:
        result = [{
            'name': section.title.strip() if section.title.strip() else 'Main',
            'entries': [e for l in section.lists() for e in _extract_entries_for_list(l)]
        } for section in wiki_text.sections]
    elif list_type == LIST_TYPE_TABLE:
        result = [{
            'name': section.title.strip() if section.title.strip() else 'Main',
            'tables': [_extract_table(t) for t in section.tables]
        } for section in wiki_text.sections]

    return result


def _extract_entries_for_list(l: wtp.WikiList) -> list:
    entries = []
    for item_text in l.items:
        plaintext, entities = wiki_parse._convert_markup(item_text)
        entries.append({
            'text': plaintext,
            'depth': l.level,
            'leaf': len(l.sublists()) == 0,
            'entities': entities
        })

    for sl in l.sublists():
        entries.extend(_extract_entries_for_list(sl))

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
            plaintext, entities = wiki_parse._convert_markup(column)
            parsed_columns.append({
                'text': plaintext,
                'entities': entities
            })
        row_data.append(parsed_columns)
    return row_data
