import wikitextparser as wtp
from wikitextparser import WikiText
from regex._regex_core import error as RegexError
import re
import impl.dbpedia.util as dbp_util
import util


LIST_TYPE_ENUM, LIST_TYPE_TABLE, LIST_TYPE_NONE = 'list_type_enum', 'list_type_table', 'list_type_none'


def parse_entries(listpage_markup: str) -> list:
    wiki_text = wtp.parse(listpage_markup)
    cleaned_wiki_text = _convert_special_enums(wiki_text)

    if _get_list_type(cleaned_wiki_text) != LIST_TYPE_ENUM:
        # TODO: implement table-lists
        return []

    return _extract_entries(cleaned_wiki_text)


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
            util.get_logger().debug('LIST-PARSE: found list without concrete type: {}'.format(wiki_text))
            return LIST_TYPE_NONE
        elif enum_entry_count > table_row_count:
            return LIST_TYPE_ENUM
        else:
            return LIST_TYPE_TABLE
    except RegexError as reg_err:
        util.get_logger().debug('LIST-PARSE: could not parse list due to exception "{}": {}'.format(reg_err, wiki_text))
        return LIST_TYPE_NONE


def _extract_entries(wiki_text: WikiText) -> list:
    entries = []
    depth = 0

    sections = wiki_text.sections
    sections_total = len(sections)
    for section_idx, section in enumerate(sections):
        section_name = section.title.strip() if section.title.strip() else 'Main'
        section_invidx = sections_total - section_idx - 1
        current_lists = section.lists()
        while current_lists:
            entries.extend([(wtp.parse(text), depth, section_name, section_idx, section_invidx) for l in current_lists for text in l.items])
            current_lists = [sl for l in current_lists for sl in l.sublists()]
            depth += 1
    return [_extract_entities(e) for e in entries]


def _extract_entities(raw_entry: tuple) -> dict:
    wiki_text, depth, section_name, section_idx, section_invidx = raw_entry
    plain_text = _convert_to_plain_text(wiki_text)

    entities = [{'uri': dbp_util.name2resource(link.target), 'text': _normalize_text(link.text or link.target)} for link in wiki_text.wikilinks]
    return {'text': plain_text, 'depth': depth, 'section_name': section_name, 'section_idx': section_idx, 'section_invidx': section_invidx, 'entities': entities}


def _convert_to_plain_text(wiki_text: WikiText) -> str:
    result = wiki_text.string
    for tag in wiki_text.tags():
        if tag._match is not None:
            result = result.replace(tag.string, _wrap_in_spaces(tag.contents))
    for link in wiki_text.external_links:
        try:
            result = result.replace(link.string, _wrap_in_spaces(link.text) if link.text else ' ')
        except AttributeError:
            result = result.replace(link.string, ' ')
    for link in wiki_text.wikilinks:
        result = result.replace(link.string, _wrap_in_spaces(link.text) if link.text else _wrap_in_spaces(link.target))
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