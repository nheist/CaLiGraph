"""Functionality for parsing Wikipedia pages from WikiText."""

from typing import Tuple, Optional, Dict, Set, List
import wikitextparser as wtp
from wikitextparser import WikiText
import impl.util.nlp as nlp_util
from impl.util.rdf import Namespace, label2name
from ..util.transformer import EntityIndex
from impl.dbpedia.util import is_entity_name
from . import wikimarkup_parser as wmp
import re
import signal
import utils
from utils import get_logger
from tqdm import tqdm
import multiprocessing as mp
from impl.dbpedia.resource import DbpResource, DbpResourceStore


LISTING_INDICATORS = ('*', '#', '{|')
VALID_ENUM_PATTERNS = (r'\#', r'\*')


class WikipediaPage:
    idx: int
    resource: DbpResource
    sections: List[dict]

    def __init__(self, resource: DbpResource, sections: List[dict]):
        self.idx = resource.idx
        self.resource = resource
        self.sections = sections

    def has_listings(self) -> bool:
        return len(self.get_enums()) > 0 or len(self.get_tables()) > 0

    def get_enums(self) -> List[list]:
        return [enum for s in self.sections for enum in s['enums']]

    def get_tables(self) -> List[dict]:
        return [table for s in self.sections for table in s['tables']]

    def get_listing_entities(self) -> Set[DbpResource]:
        dbr = DbpResourceStore.instance()
        entity_indices = {ent['idx'] for enum in self.get_enums() for entry in enum for ent in entry['entities']} |\
                         {ent['idx'] for table in self.get_tables() for row in table['data'] for cell in row for ent in cell['entities']}
        return {dbr.get_resource_by_idx(idx) for idx in entity_indices if idx != EntityIndex.NEW_ENTITY.value}

    def discard_listings_without_seen_entities(self):
        # discard listings that either do not have any subject entities or only unseen subject entities
        for s in self.sections:
            s['enums'] = [enum for enum in s['enums'] if any('subject_entity' in entry and entry['subject_entity']['idx'] != EntityIndex.NEW_ENTITY.value for entry in enum)]
            s['tables'] = [t for t in s['tables'] if any('subject_entity' in cell and cell['subject_entity']['idx'] != EntityIndex.NEW_ENTITY.value for row in t['data'] for cell in row)]

    def get_subject_entity_indices(self) -> Set[int]:
        return {entry['subject_entity']['idx'] for enum in self.get_enums() for entry in enum if 'subject_entity' in entry} |\
               {cell['subject_entity']['idx'] for table in self.get_tables() for row in table['data'] for cell in row if 'subject_entity' in cell}

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['resource']  # do not persist DbpResource directly, but recover it from idx
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.resource = DbpResourceStore.instance().get_resource_by_idx(self.idx)


def _parse_pages(pages_markup: Dict[DbpResource, str]) -> List[WikipediaPage]:
    # warm up caches before going into multiprocessing
    dbr = DbpResourceStore.instance()
    res = dbr.get_resource_by_idx(0)
    dbr.get_label(res)
    dbr.resolve_redirect(res)

    wikipedia_pages = []
    with mp.Pool(processes=utils.get_config('max_cpus')) as pool:
        for wp in tqdm(pool.imap_unordered(_parse_page_with_timeout, pages_markup.items(), chunksize=1000), total=len(pages_markup), desc='wikipedia/page_parser: Parsing pages'):
            if wp.has_listings():
                wikipedia_pages.append(wp)
            pages_markup[wp.resource] = ''  # discard markup after parsing to free memory
    return wikipedia_pages


def _parse_page_with_timeout(resource_and_markup: Tuple[DbpResource, str]) -> WikipediaPage:
    """Return the parsed wikipedia page (with empty content, if parsing has timed out)"""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5 * 60)  # timeout of 5 minutes per page

    resource = resource_and_markup[0]
    try:
        wp = _parse_page(resource_and_markup)
        signal.alarm(0)  # reset alarm as parsing was successful
        return wp
    except Exception as e:
        if type(e) == KeyboardInterrupt:
            raise e
        get_logger().error(f'Failed to parse page {resource.name}: {e}')
        return WikipediaPage(resource, [])


def _parse_page(resource_and_markup: Tuple[DbpResource, str]) -> WikipediaPage:
    resource, page_markup = resource_and_markup
    wp = WikipediaPage(resource, [])
    if not any(indicator in page_markup for indicator in LISTING_INDICATORS):
        return wp  # early return if page contains no listings at all
    # prepare markup for parsing
    page_markup = page_markup.replace('&nbsp;', ' ')  # replace html whitespaces
    page_markup = page_markup.replace('<br />', ' ')  # replace html line breaks
    page_markup = page_markup.replace('<br/>', ' ')
    page_markup = page_markup.replace('<br>', ' ')
    page_markup = re.sub(r'<ref>.*?</ref>', '', page_markup)  # remove ref markers
    page_markup = re.sub(r'<ref[^>]*?/>', '', page_markup)
    page_markup = re.sub(r"'{2,}", '', page_markup)  # remove bold and italic markers
    # early return if page is not useful
    wiki_text = wtp.parse(page_markup)
    if not _is_page_useful(wiki_text):
        return wp
    # clean and expand markup
    cleaned_wiki_text = _convert_special_enums(wiki_text)
    cleaned_wiki_text = _remove_enums_within_tables(cleaned_wiki_text)
    if not _is_page_useful(cleaned_wiki_text):
        return wp
    cleaned_wiki_text = _expand_wikilinks(cleaned_wiki_text, resource)
    # extract data from sections
    wp.sections = _extract_sections(cleaned_wiki_text)
    return wp


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


def _expand_wikilinks(wiki_text: WikiText, resource: DbpResource) -> WikiText:
    text_to_wikilink = {wl.text or wl.target: wl.string for wl in wiki_text.wikilinks if is_entity_name(label2name(wl.target))}
    text_to_wikilink[resource.get_label()] = f'[[{resource.name}]]'  # replace mentions of the page title with a link to it
    # discard wikilinks that have text which is fully contained in other wikilinks to avoid nested wikilinks
    wikilinks_words = [_get_alphanum_words(wl.text) for wl in wiki_text.wikilinks if wl.text] + [_get_alphanum_words(wl.target) for wl in wiki_text.wikilinks if wl.target]
    # if the words of a wikilink are a proper subset of the words of another wikilink, we discard it
    # (if the sets are equal, then we are most likely looking at the words of the entity itself; this case is handled in the look-ahead of the regex)
    text_to_wikilink = {text: wl for text, wl in text_to_wikilink.items() if not any(_get_alphanum_words(text) < wl_words for wl_words in wikilinks_words)}
    # replace text with wikilinks
    pattern_to_wikilink = {r'(?<![|\[])\b' + re.escape(text) + r'\b(?![|\]])': wl for text, wl in text_to_wikilink.items()}
    regex = re.compile("|".join(pattern_to_wikilink.keys()))
    try:
        # For each match, look up the corresponding value in the dictionary
        return wtp.parse(regex.sub(lambda match: text_to_wikilink[match.group(0)], wiki_text.string))
    except Exception as e:
        if type(e) in [KeyboardInterrupt, ParsingTimeoutException]:
            raise e
        return wiki_text


def _get_alphanum_words(text: str) -> Set[str]:
    return set(re.sub(r'[^a-zA-Z0-9_ ]+', '', text).split())


def _extract_sections(wiki_text: WikiText) -> list:
    sections = []
    for section_idx, section in enumerate(wiki_text.get_sections(include_subsections=False)):
        enums = [_extract_enum(l) for l in section.get_lists(VALID_ENUM_PATTERNS)]
        tables = [_extract_table(t) for t in section.get_tables()]
        sections.append({
            'index': section_idx,
            'name': section.title.strip() if section.title and section.title.strip() else 'Main',
            'level': section.level,
            'enums': [e for e in enums if len(e) >= 2],
            'tables': [t for t in tables if t]
        })
    return sections


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
            parsed_cells.append({'text': plaintext, 'entities': entities})
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
    parsed_text = wtp.parse(wiki_text)
    parsed_text = _remove_file_wikilinks(parsed_text)
    parsed_text = _convert_sortname_templates(parsed_text)

    plain_text = wmp.wikitext_to_plaintext(parsed_text).strip()

    # extract wikilink-entities with correct positions in plain text
    entities = []
    current_index = 0
    for w in parsed_text.wikilinks:
        # retrieve entity text and remove html tags
        text = (w.text or w.target).strip()
        text = nlp_util.remove_bracket_content(text, bracket_type='<')
        if w.target.startswith((Namespace.PREFIX_FILE.value, Namespace.PREFIX_IMAGE.value)):
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
        entity_idx = wmp.get_resource_idx_for_wikilink(w)
        if entity_idx is not None:
            entities.append({'start': entity_start_index, 'text': text, 'idx': entity_idx})
    return plain_text, entities


def _remove_file_wikilinks(parsed_text: wtp.WikiText) -> wtp.WikiText:
    """Remove wikilinks to files or images."""
    for wl in reversed(parsed_text.wikilinks):
        if wl.string.startswith(('[[File:', '[[Image:')):
            parsed_text[slice(*wl.span)] = ''
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
