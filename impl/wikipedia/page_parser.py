"""Functionality for parsing Wikipedia pages from WikiText."""

from typing import Tuple, Optional
import wikitextparser as wtp
from wikitextparser import WikiText
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.util.nlp as nlp_util
from . import wikimarkup_parser as wmp
import re
import signal
import utils
from utils import get_logger
from tqdm import tqdm
import multiprocessing as mp
from enum import Enum


LISTING_INDICATORS = ('*', '#', '{|')
VALID_ENUM_PATTERNS = (r'\#', r'\*')


class PageType(Enum):
    ENUM = 'enumeration'
    TABLE = 'table'


def _parse_pages(pages_markup) -> dict:
    dbp_store.resolve_redirect('')  # make sure redirects are initialised before going into multiprocessing

    parsed_pages = {}
    with mp.Pool(processes=utils.get_config('max_cpus')) as pool:
        for r, parsed in tqdm(pool.imap_unordered(_parse_page_with_timeout, pages_markup.items(), chunksize=2000), total=len(pages_markup), desc='Parsing pages'):
            if parsed:
                parsed_pages[r] = parsed
            del pages_markup[r]  # discard markup after parsing to free space
    return parsed_pages


def _parse_page_with_timeout(resource_and_markup: tuple) -> tuple:
    """Return a single parsed page in the following hierarchical structure:

    Sections > Enums > Entries > Entities
    Sections > Tables > Rows > Columns > Entities
    """
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5 * 60)  # timeout of 5 minutes per page

    resource = resource_and_markup[0]
    try:
        result = _parse_page(resource_and_markup)
        signal.alarm(0)  # reset alarm as parsing was successful
        return result
    except Exception as e:
        if type(e) == KeyboardInterrupt:
            raise e
        get_logger().error(f'Failed to parse page {resource}: {e}')
        return resource, None


def _parse_page(resource_and_markup: tuple) -> tuple:
    resource, page_markup = resource_and_markup
    if dbp_util.is_file_resource(resource):
        return resource, None  # discard files and images
    if not any(indicator in page_markup for indicator in LISTING_INDICATORS):
        return resource, None  # early return of 'None' if page contains no listings at all

    # prepare markup for parsing
    page_markup = page_markup.replace('&nbsp;', ' ')  # replace html whitespaces
    page_markup = page_markup.replace('<br/>', '\n')  # replace html line breaks
    page_markup = page_markup.replace('<br>', '\n')  # replace html line breaks
    page_markup = re.sub(r'<ref>.*?</ref>', '', page_markup)
    page_markup = re.sub(r'<ref[^>]*?/>', '', page_markup)
    page_markup = re.sub(r"'{2,}", '', page_markup)  # remove bold and italic markers

    wiki_text = wtp.parse(page_markup)
    if not _is_page_useful(wiki_text):
        return resource, None

    cleaned_wiki_text = _convert_special_enums(wiki_text)
    cleaned_wiki_text = _remove_enums_within_tables(cleaned_wiki_text)
    if not _is_page_useful(cleaned_wiki_text):
        return resource, None

    # expand wikilinks
    resource_name = dbp_util.resource2name(resource)
    cleaned_wiki_text = _expand_wikilinks(cleaned_wiki_text, resource_name)

    # extract data from sections
    sections = _extract_sections(cleaned_wiki_text)
    types = set()
    if any(len(s['enums']) > 0 for s in sections):
        types.add(PageType.ENUM)
    if any(len(s['tables']) > 0 for s in sections):
        types.add(PageType.TABLE)
    if not types:
        return resource, None  # ignore pages without useful lists
    return resource, {'sections': sections, 'types': types}


def _is_page_useful(wiki_text: WikiText) -> bool:
    # ignore pages without any lists and pages with very small lists (e.g. redirect pages have a list with length of 1)
    return len(wiki_text.get_lists(VALID_ENUM_PATTERNS)) + len(wiki_text.get_tables()) > 0


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


def _expand_wikilinks(wiki_text: WikiText, resource_name: str) -> WikiText:
    invalid_wikilink_prefixes = ['File:', 'Image:', 'Category:', 'List of']
    text_to_wikilink = {wl.text or wl.target: wl.string for wl in wiki_text.wikilinks if not any(wl.target.startswith(prefix) for prefix in invalid_wikilink_prefixes)}
    text_to_wikilink[resource_name] = f'[[{resource_name}]]'  # replace mentions of the page title with a link to it
    pattern_to_wikilink = {r'(?<![|\[])\b' + re.escape(text) + r'\b(?![|\]])': wl for text, wl in text_to_wikilink.items()}
    regex = re.compile("|".join(pattern_to_wikilink.keys()))
    try:
        # For each match, look up the corresponding value in the dictionary
        return wtp.parse(regex.sub(lambda match: text_to_wikilink[match.group(0)], wiki_text.string))
    except Exception as e:
        if type(e) in [KeyboardInterrupt, ParsingTimeoutException]:
            raise e
        return wiki_text


def _extract_sections(wiki_text: WikiText) -> list:
    sections = []
    for section_idx, section in enumerate(wiki_text.get_sections(include_subsections=False)):
        #markup_without_lists = _remove_listing_markup(section)
        #text, entities = _convert_markup(markup_without_lists)
        enums = [_extract_enum(l) for l in section.get_lists(VALID_ENUM_PATTERNS)]
        tables = [_extract_table(t) for t in section.get_tables()]
        sections.append({
            'index': section_idx,
            'name': section.title.strip() if section.title and section.title.strip() else 'Main',
            'level': section.level,
            #'text': text,
            #'entities': entities,
            'enums': [e for e in enums if len(e) >= 2],
            'tables': [t for t in tables if t]
        })
    return sections


def _remove_listing_markup(wiki_text: WikiText) -> str:
    result = wiki_text.string
    for indicator in LISTING_INDICATORS:
        if indicator in result:
            result = result[:result.index(indicator)]
    return result


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


def _extract_table(table: wtp.Table) -> Optional[dict]:
    row_header = []
    row_data = []
    try:
        rows = table.data(strip=True, span=True)
        cells = table.cells(span=True)
        rows_with_spans = table.data(strip=True, span=False)
    except Exception as e:
        if type(e) in [KeyboardInterrupt, ParsingTimeoutException]:
            raise e
        return None
    for row_idx, row in enumerate(rows):
        if len(row) < 2 or len(row) > 100:
            # ignore tables with only one or more than 100 columns (likely irrelevant or markup error)
            return None
        parsed_cells = []
        for cell in row:
            plaintext, entities = _convert_markup(str(cell))
            parsed_cells.append({
                'text': plaintext,
                'entities': entities
            })
        if _is_header_row(cells, row_idx):
            row_header = parsed_cells
        else:
            if len(rows_with_spans) > row_idx and len(row) == len(rows_with_spans[row_idx]):
                # only use rows that are not influenced by row-/colspan
                row_data.append(parsed_cells)
    if len(row_data) < 2:
        return None  # ignore tables with less than 2 data rows
    return {'header': row_header, 'data': row_data}


def _is_header_row(cells, row_idx: int) -> bool:
    try:
        return row_idx == 0 or any(c and c.is_header for c in cells[row_idx])
    except IndexError:
        return False  # fallback if wtp can't parse the table correctly


def _convert_markup(wiki_text: str) -> Tuple[str, list]:
    # preprocess markup text
    parsed_text = _remove_inner_wikilinks(wtp.parse(wiki_text))
    parsed_text = _convert_sortname_templates(parsed_text)

    plain_text = wmp.wikitext_to_plaintext(parsed_text).strip()

    # extract wikilink-entities with correct positions in plain text
    entities = []
    current_index = 0
    for w in parsed_text.wikilinks:
        # retrieve entity text and remove html tags
        text = (w.text or w.target).strip()
        text = nlp_util.remove_bracket_content(text, bracket_type='<')
        if w.target.startswith(('File:', 'Image:')):
            continue  # ignore files and images
        if '|' in text:  # deal with invalid markup in wikilinks
            text = text[text.rindex('|')+1:].strip()
        if not text:
            continue  # skip entity with empty text

        # retrieve entity position
        if text not in plain_text[current_index:]:
            continue  # skip entity with a text that can not be located
        entity_start_index = current_index + plain_text[current_index:].index(text)
        current_index = entity_start_index + len(text)
        entity_name = wmp.get_entity_for_wikilink(w)
        if entity_name:
            entities.append({'idx': entity_start_index, 'text': text, 'name': entity_name})
    return plain_text, entities


def _remove_inner_wikilinks(parsed_text: wtp.WikiText) -> wtp.WikiText:
    """Remove inner wikilinks that might have been created by wikilink-expansion."""
    for wikilink in parsed_text.wikilinks:
        if not wikilink or not wikilink.string.startswith('[['):
            continue
        for wl in reversed(wikilink.wikilinks):
            parsed_text[slice(*wl.span)] = wl.text or wl.target
    return parsed_text


def _convert_sortname_templates(parsed_text: wtp.WikiText) -> wtp.WikiText:
    """Convert Sortname template (typically found in tables) into a simple wikilink.

    Documentation of Sortname template: https://en.wikipedia.org/wiki/Template:Sortname
    """
    for t in parsed_text.templates:
        if not t.string.startswith('{{') or t.normal_name(capitalize=True) != 'Sortname':
            continue
        text = (t.get_arg('1').value + ' ' + t.get_arg('2').value).strip()
        if t.has_arg('nolink'):
            result = text
        else:
            if t.has_arg('3'):
                link = t.get_arg('3').value
                result = f'[[{link}|{text}]]'
            else:
                result = f'[[{text}]]'
        parsed_text[slice(*t.span)] = result
    return parsed_text


# define functionality for parsing timeouts


class ParsingTimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ParsingTimeoutException('Parsing timeout.')
