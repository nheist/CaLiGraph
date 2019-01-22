from collections import namedtuple
import wikitextparser as wtp
from wikitextparser import WikiText
from _regex_core import error as RegexError
import re
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import util
import impl.util.nlp as nlp_util


LIST_TYPE_ENUM, LIST_TYPE_TABLE, LIST_TYPE_NONE = 'list_type_enum', 'list_type_table', 'list_type_none'
LABEL_RESOURCE, LABEL_OTHER, LABEL_MISSING = 1, 0, -1
NOISE_SECTIONS = ['See also', 'References', 'External links', 'Sources and external links']


ListEntry = namedtuple('ListEntry', 'words wikitext depth section_name section_pos section_first section_last types')
ListEntryWord = namedtuple('ListEntryWord', 'word pos ner label')


def parse_entries(listpage_markup: str, dbp_types: set) -> list:
    wiki_text = wtp.parse(listpage_markup)
    cleaned_wiki_text = _convert_special_enums(wiki_text)

    if _get_list_type(cleaned_wiki_text) != LIST_TYPE_ENUM:
        # TODO: implement table-lists
        return []

    return [_finalize_entry(re, dbp_types) for re in _extract_raw_entries(cleaned_wiki_text)]


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
    except RegexError as re:
        util.get_logger().debug('LIST-PARSE: could not parse list due to exception "{}": {}'.format(re, wiki_text))
        return LIST_TYPE_NONE


def _extract_raw_entries(wiki_text: WikiText) -> list:
    """Return list page entries as Tuple(text: WikiText, depth: int)"""
    entries = []
    depth = 0

    sections = wiki_text.sections
    sections_total = len(sections)
    for section_idx, section in enumerate(sections):
        section_name = section.title.strip() if section.title.strip() else 'Main'
        section_first = section_idx == 0
        section_last = section_idx + 1 == sections_total
        current_lists = section.lists()
        while current_lists:
            entries.extend([(wtp.parse(text), depth, section_name, section_idx, section_first, section_last) for l in current_lists for text in l.items])
            current_lists = [sl for l in current_lists for sl in l.sublists()]
            depth += 1
    return entries


def _finalize_entry(raw_entry: tuple, dbp_types: set) -> ListEntry:
    wiki_text, depth, section_name, section_pos, section_first, section_last = raw_entry

    plain_text = _convert_to_plain_text(wiki_text)
    if plain_text.strip():
        tokens = _tag_text(plain_text)
        labels = _get_labels(wiki_text, [t['word'] for t in tokens], dbp_types)
        words = [ListEntryWord(word=token['word'], pos=token['pos'], ner=token['ner'], label=labels[idx]) for idx, token in enumerate(tokens)]
    else:
        words = []

    return ListEntry(words=words, wikitext=wiki_text, depth=depth, section_name=section_name, section_pos=section_pos, section_first=section_first, section_last=section_last, types=dbp_types)


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

    result = re.sub("'{2,}", '', result)
    result = re.sub(' +', ' ', result)
    return result.strip()


def _wrap_in_spaces(word: str) -> str:
    return ' ' + word + ' '


def _get_labels(wiki_text: WikiText, tokenized_text: list, dbp_types: set) -> list:
    # locate resources that belong to the list
    resources = []
    independent_dbp_types = dbp_store.get_independent_types(dbp_types)
    for resource_candidate in wiki_text.wikilinks:
        target = resource_candidate.target
        text = resource_candidate.text if resource_candidate.text else target
        resource_types = dbp_store.get_types(dbp_util.name2resource(target))
        if not independent_dbp_types.difference(resource_types):
            resources.append(_tokenize_text(text))

    # return undefined labels if no resource could be found
    if not resources:
        return [LABEL_MISSING] * len(tokenized_text)

    # return boolean labels for every tokenized word
    labels = []
    for i, word in enumerate(tokenized_text):
        if len(labels) > i:
            continue

        words_found = 0
        for res in resources:
            if word == res[0] and res == tokenized_text[i:i+len(res)]:
                words_found = len(res)

        if words_found == 0:
            labels.append(LABEL_OTHER)
        else:
            labels.extend([LABEL_RESOURCE] * words_found)

    return labels


def _tokenize_text(text: str) -> list:
    return [t.text for t in nlp_util.parse(text, skip_cache=True)]


def _tag_text(text: str) -> list:
    return [{'word': t.text, 'pos': t.tag_, 'ner': t.ent_type_ if t.ent_type_ else 'O'} for t in nlp_util.parse(text, skip_cache=True)]
