"""Process the Wikipedia XML dump and return articles, categories, and templates."""

import utils
from utils import get_logger
from lxml import etree
import bz2
import impl.dbpedia.util as dbp_util


def _parse_raw_markup_from_xml() -> dict:
    get_logger().info('Parsing raw markup from XML dump..')
    parser = etree.XMLParser(target=WikiPageParser())
    with bz2.open(utils.get_data_file('files.wikipedia.pages')) as dbp_pages_file:
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
            if self.processed_pages % 100000 == 0:
                get_logger().debug(f'Parsed markup of {self.processed_pages} pages.')

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
        return self.namespace in ['0', '10', '14']  # 0 = Page, 10 = Template, 14 = Category
