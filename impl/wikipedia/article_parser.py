"""Functionality for parsing Wikipedia pages from WikiText."""

import wikitextparser as wtp
from wikitextparser import WikiText
from typing import Tuple, Optional
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.util.nlp as nlp_util
import impl.util.string as str_util
import re
import signal
import utils
from tqdm import tqdm
import multiprocessing as mp


LISTING_INDICATORS = ('*', '#', '{|')
VALID_ENUM_PATTERNS = (r'\#', r'\*')
ARTICLE_TYPE_ENUM, ARTICLE_TYPE_TABLE = 'enum', 'table'


def _parse_articles(articles_markup) -> dict:
    with mp.Pool(processes=round(utils.get_config('max_cpus') / 2)) as pool:
        parsed_articles = {r: parsed for r, parsed in tqdm(pool.imap_unordered(_parse_article_with_timeout, articles_markup.items(), chunksize=2000), total=len(articles_markup)) if parsed}
    return parsed_articles


def _parse_article_with_timeout(resource_and_markup: tuple) -> tuple:
    """Return a single parsed article in the following hierarchical structure:

    Sections > Enums > Entries > Entities
    Sections > Tables > Rows > Columns > Entities
    """
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5 * 60)  # timeout of 5 minutes per page

    resource = resource_and_markup[0]
    try:
        result = _parse_article(resource_and_markup)
        signal.alarm(0)  # reset alarm as parsing was successful
        return result
    except Exception as e:
        utils.get_logger().error(f'WIKIPEDIA/ARTICLES: Failed to parse page {resource}: {e}')
        return resource, None


def _parse_article(resource_and_markup: tuple) -> tuple:
    resource, page_markup = resource_and_markup
    if not any(indicator in page_markup for indicator in LISTING_INDICATORS):
        return resource, None  # early return of 'None' if page contains no listings at all

    # prepare markup for parsing
    page_markup = page_markup.replace('&nbsp;', ' ')  # replace html whitespaces
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
        types.add(ARTICLE_TYPE_ENUM)
    if any(len(s['tables']) > 0 for s in sections):
        types.add(ARTICLE_TYPE_TABLE)
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
        if type(e) == ParsingTimeoutException:
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
        if type(e) == ParsingTimeoutException:
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
    parsed_text = wtp.parse(wiki_text)
    plain_text = _wikitext_to_plaintext(parsed_text).strip()

    # extract wikilink-entities with correct positions in plain text
    entities = []
    current_entity_index = 0
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
        if text not in plain_text[current_entity_index:]:
            continue  # skip entity with a text that can not be located
        entity_position = current_entity_index + plain_text[current_entity_index:].index(text)
        current_entity_index = entity_position + len(text)
        entity_name = _convert_target_to_name(w.target)
        if entity_name:
            entities.append({'idx': entity_position, 'text': text, 'name': entity_name})
    return plain_text, entities


def _wikitext_to_plaintext(parsed_text: wtp.WikiText) -> str:
    # bolds and italics are already removed during preprocessing to reduce runtime
    result = parsed_text.plain_text(replace_bolds_and_italics=False).strip(" '\t\n")
    result = re.sub(r'\n+', '\n', result)
    result = re.sub(r' +', ' ', result)
    return result


def _convert_target_to_name(link_target: str) -> Optional[str]:
    if not link_target:
        return None
    link_target = _remove_language_tag(link_target.strip())
    resource_uri = dbp_util.name2resource(str_util.capitalize(link_target))
    redirected_uri = dbp_store.resolve_spelling_redirect(resource_uri)
    if dbp_store.is_possible_resource(redirected_uri) and '#' not in redirected_uri:
        # return redirected uri only if it is an own Wikipedia article and it does not point to an article section
        final_uri = redirected_uri
    else:
        final_uri = resource_uri
    return dbp_util.resource2name(final_uri)


def _remove_language_tag(link_target: str) -> str:
    if link_target[0] != ':':
        return link_target
    if len(link_target) < 4 or link_target[3] != ':':
        return link_target[1:]
    return link_target[4:]


# define functionality for parsing timeouts


class ParsingTimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ParsingTimeoutException('Parsing timeout.')
