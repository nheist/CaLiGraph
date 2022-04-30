import wikitextparser as wtp
from typing import Union, Optional
import re
import impl.util.string as str_util
from impl.dbpedia.resource import DbpEntity, DbpResourceStore
from impl.dbpedia.util import is_entity_name
from impl.util.rdf import label2name


def wikitext_to_plaintext(text: Union[str, wtp.WikiText]) -> str:
    try:
        parsed_text = wtp.parse(text) if isinstance(text, str) else text
        # bolds and italics are already removed during preprocessing to reduce runtime
        result = parsed_text.plain_text(replace_bolds_and_italics=False).strip(" '\t\n")
        result = re.sub(r'\n+', '\n', result)
        result = re.sub(r' +', ' ', result)
        return result
    except (AttributeError, IndexError):
        return str(text)


def get_first_wikilink_entity(text: Union[str, wtp.WikiText]) -> Optional[int]:
    dbr = DbpResourceStore.instance()
    try:
        parsed_text = wtp.parse(text) if type(text) == str else text
        for wl in parsed_text.wikilinks:
            res_idx = get_resource_idx_for_wikilink(wl)
            if not dbr.has_resource_with_idx(res_idx):
                continue
            res = dbr.get_resource_by_idx(res_idx)
            if not isinstance(res, DbpEntity):
                continue
            return res_idx
        return None
    except (AttributeError, IndexError):
        return None


def get_resource_idx_for_wikilink(wikilink: wtp.WikiLink) -> Optional[int]:
    if not wikilink.target:
        return None
    dbr = DbpResourceStore.instance()
    res_name = label2name(str_util.capitalize(_remove_language_tag(wikilink.target.strip())))
    if dbr.has_resource_with_name(res_name):
        res = dbr.get_resource_by_name(res_name)
        if not res.is_meta:
            return res.idx
        redirected_res = dbr.resolve_spelling_redirect(res)
        if not redirected_res.is_meta:
            return redirected_res.idx
    return -1 if is_entity_name(res_name) else None


def _remove_language_tag(link_target: str) -> str:
    if link_target[0] != ':':
        return link_target
    if len(link_target) < 4 or link_target[3] != ':':
        return link_target[1:]
    return link_target[4:]
