"""Retrieve page information of articles in DBpedia."""

import util
from lxml import etree
import bz2
import impl.dbpedia.util as dbp_util
import impl.util.wiki_parse as wiki_parse
from collections import defaultdict
import numpy as np
import multiprocessing as mp


def get_all_parsed_pages() -> dict:
    return defaultdict(lambda: None, util.load_or_create_cache('dbpedia_page_parsed', _parse_pages))


def _parse_pages() -> dict:
    params = [chunk for chunk in np.array_split(list(get_all_pages_markup().items()), 5)]  # todo: set chunks to max_cpus if working
    with mp.Pool(processes=len(params)) as pool:
        results = pool.map(_parse_page_chunk, params)
    return {res: parsed_page for chunk_result in results for res, parsed_page in chunk_result.items()}


def _parse_page_chunk(chunk_of_pages: list) -> dict:
    parsed_pages = {}
    for idx, (resource, markup) in enumerate(chunk_of_pages):
        if idx % 10000 == 0:
            util.get_logger().debug(f'DBPEDIA/PAGES ({mp.current_process().name}): Parsed {idx}/{len(chunk_of_pages)} pages.')
        try:
            parsed_pages[resource] = wiki_parse.parse_page(markup)
        except Exception as e:
            util.get_logger().error(f'DBPEDIA/PAGES ({mp.current_process().name}): Failed to parse page {resource}: {e}')
    return parsed_pages


def get_all_pages_markup() -> dict:
    return defaultdict(str, util.load_or_create_cache('dbpedia_page_markup', _fetch_page_markup))


def _fetch_page_markup() -> dict:
    util.get_logger().info('CACHE: Parsing page markup')
    parser = etree.XMLParser(target=WikiPageParser())
    with bz2.open(util.get_data_file('files.dbpedia.pages')) as dbp_pages_file:
        page_markup = etree.parse(dbp_pages_file, parser)
        return {dbp_util.name2resource(p): markup for p, markup in page_markup.items()}


class WikiPageParser:
    """Parse WikiText as stream and return content based on page markers (only for simple article pages)."""
    def __init__(self):
        self.processed_pages = 0
        self.page_markup = {}
        self.title = None
        self.namespace = None
        self.tag_content = ''

    def start(self, tag, _):
        if tag.endswith('}page'):
            self.title = None
            self.namespace = None
            self.processed_pages += 1
            if self.processed_pages % 1000 == 0:
                util.get_logger().debug(f'PAGE MARKUP: Processed {self.processed_pages} pages.')

    def end(self, tag):
        if tag.endswith('}title'):
            self.title = self.tag_content.strip()
        elif tag.endswith('}ns'):
            self.namespace = self.tag_content.strip()
        elif tag.endswith('}text') and self._valid_page():
            self.page_markup[self.title] = self.tag_content.strip()
        self.tag_content = ''

    def data(self, chars):
        self.tag_content += chars

    def close(self) -> dict:
        return self.page_markup

    def _valid_page(self) -> bool:
        return self.namespace == '0' and not self.title.startswith('List of ')
