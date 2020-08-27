"""Retrieve page information of articles in DBpedia."""

import util
from lxml import etree
import bz2
import impl.dbpedia.util as dbp_util
import impl.util.wiki_parse as wiki_parse
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm


def get_all_parsed_pages() -> dict:
    return defaultdict(lambda: None, util.load_or_create_cache('dbpedia_page_parsed', _parse_pages))


def _parse_pages() -> dict:
    with mp.Pool(processes=util.get_config('max_cpus')) as pool:
        results = tqdm(pool.imap(_parse_page, get_all_pages_markup().items(), chunksize=2000))
    return {resource: parsed_page for resource, parsed_page in results if parsed_page}


def _parse_page(resource_and_markup: tuple) -> tuple:
    resource, markup = resource_and_markup
    try:
        return resource, wiki_parse.parse_page(markup)
    except Exception as e:
        util.get_logger().error(f'DBPEDIA/PAGES ({mp.current_process().name}): Failed to parse page {resource}: {e}')
    return resource, None


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
        return self.namespace == '0'
