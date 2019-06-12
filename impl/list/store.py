from . import util as list_util
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import util
from lxml import etree
from collections import defaultdict
import bz2


def get_listpages() -> set:
    global __LISTPAGES__
    if '__LISTPAGES__' not in globals():
        __LISTPAGES__ = {dbp_store.resolve_redirect(lp) for lp in get_listpages_with_redirects() if list_util.is_listpage(dbp_store.resolve_redirect(lp))}

    return __LISTPAGES__


def get_listpages_with_redirects() -> set:
    global __LISTPAGES_WITH_REDIRECTS__
    if '__LISTPAGES_WITH_REDIRECTS__' not in globals():
        initializer = lambda: {res for res in dbp_store.get_resources() if list_util.is_listpage(res)}
        __LISTPAGES_WITH_REDIRECTS__ = util.load_or_create_cache('dbpedia_listpages', initializer)

    return __LISTPAGES_WITH_REDIRECTS__


def get_listpage_markup(listpage: str) -> str:
    global __LISTPAGE_MARKUP__
    if '__LISTPAGE_MARKUP__' not in globals():
        __LISTPAGE_MARKUP__ = defaultdict(lambda: '', util.load_or_create_cache('dbpedia_listpage_markup', _fetch_listpage_markup))

    return __LISTPAGE_MARKUP__[listpage]


def _fetch_listpage_markup():
    util.get_logger().info('CACHE: Parsing listpage markup')
    parser = etree.XMLParser(target=WikiListpageParser())
    with bz2.open(util.get_data_file('files.dbpedia.pages')) as dbp_pages_file:
        list_markup = etree.parse(dbp_pages_file, parser)
        return {dbp_util.name2resource(lp): markup for lp, markup in list_markup.items()}


class WikiListpageParser:
    def __init__(self):
        self.processed_pages = 0
        self.list_markup = {}
        self.title = None
        self.namespace = None
        self.tag_content = ''

    def start(self, tag, _):
        if tag.endswith('}page'):
            self.title = None
            self.namespace = None
            self.processed_pages += 1

    def end(self, tag):
        if tag.endswith('}title'):
            self.title = self.tag_content.strip()
        elif tag.endswith('}ns'):
            self.namespace = self.tag_content.strip()
        elif tag.endswith('}text') and self._valid_page():
            self.list_markup[self.title] = self.tag_content.strip()
        self.tag_content = ''

    def data(self, chars):
        self.tag_content += chars

    def close(self) -> dict:
        return self.list_markup

    def _valid_page(self) -> bool:
        return self.namespace == '0' and self.title.startswith('List of ')
