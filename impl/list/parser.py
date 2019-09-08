import wikitextparser as wtp
from wikitextparser import WikiText
from typing import Tuple, Optional
from regex._regex_core import error as RegexError
import re
import impl.dbpedia.util as dbp_util
import util
import impl.list.store as list_store


LIST_TYPE_ENUM, LIST_TYPE_TABLE, LIST_TYPE_NONE = 'list_type_enum', 'list_type_table', 'list_type_none'


def get_parsed_listpages() -> dict:
    global __PARSED_LISTPAGES__
    if '__PARSED_LISTPAGES__' not in globals():
        __PARSED_LISTPAGES__ = util.load_or_create_cache('dbpedia_listpage_parsed', _compute_parsed_listpages)

    return __PARSED_LISTPAGES__


def _compute_parsed_listpages() -> dict:
    return {lp: _parse_listpage(lp, list_store.get_listpage_markup(lp)) for lp in list_store.get_listpages()}


def _parse_listpage(listpage_uri: str, listpage_markup: str) -> dict:
    wiki_text = wtp.parse(listpage_markup)
    cleaned_wiki_text = _convert_special_enums(wiki_text)

    list_type = _get_list_type(cleaned_wiki_text)
    result = {
        'uri': listpage_uri,
        'type': list_type
    }

    if list_type != LIST_TYPE_NONE:
        result['sections'] = _extract_sections(list_type, cleaned_wiki_text)

    return result


def _convert_special_enums(wiki_text: WikiText) -> WikiText:
    result = wiki_text.string
    enum_templates = [t for t in wiki_text.templates if t.name == 'columns-list']
    for et in enum_templates:
        actual_list = et.get_arg('1')
        result = result.replace(et.string, actual_list.string[1:] if actual_list else '')
    return wtp.parse(result)


def _get_list_type(wiki_text: WikiText) -> str:
    try:
        enum_entry_count = len([entry for enum in wiki_text.lists(pattern=r'\*+') for entry in enum.items])
        table_row_count = len([row for table in wiki_text.tables for row in table.data(span=False)])

        if not enum_entry_count and not table_row_count:
            #util.get_logger().debug('LIST-PARSE: found list without concrete type: {}'.format(wiki_text))
            return LIST_TYPE_NONE
        elif enum_entry_count > table_row_count:
            return LIST_TYPE_ENUM
        else:
            return LIST_TYPE_TABLE
    except RegexError as reg_err:
        #util.get_logger().debug('LIST-PARSE: could not parse list due to exception "{}": {}'.format(reg_err, wiki_text))
        return LIST_TYPE_NONE


def _extract_sections(list_type: str, wiki_text: WikiText) -> list:
    result = []

    if list_type == LIST_TYPE_NONE:
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
        plaintext, entities = _convert_markup(item_text)
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
    rows = []
    for row in table.data(span=False):
        parsed_columns = []
        for column in row:
            plaintext, entities = _convert_markup(column)
            parsed_columns.append({
                'text': plaintext,
                'entities': entities
            })
        rows.append(parsed_columns)
    return rows


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
        'uri': dbp_util.name2resource(link_target),
        'text': link_text,
        'idx': len(pre)
    }


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
    return ' ' + word + ' '


def _normalize_text(text: str) -> str:
    text = re.sub("'{2,}", '', text)
    text = re.sub(' +', ' ', text)
    return text.strip()
