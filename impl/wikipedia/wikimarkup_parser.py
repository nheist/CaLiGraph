import wikitextparser as wtp
from typing import Union, Optional
import re
from impl.dbpedia.resource import DbpResourceStore
import impl.util.string as str_util
from impl.util.rdf import Namespace, label2name
import impl.util.nlp as nlp_util


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


def get_first_wikilink_resource(text: str) -> Optional[str]:
    try:
        for wl in wtp.parse(text).wikilinks:
            res = get_resource_name_for_wikilink(wl)
            if res is None:
                continue
            return res
        return None
    except (AttributeError, IndexError):
        return None


def get_label_for_wikilink(wikilink: wtp.WikiLink) -> Optional[str]:
    text = wikilink.text or wikilink.target
    if not text:
        return None
    text = nlp_util.remove_bracket_content(text.strip(), bracket_type='<')
    if wikilink.target.startswith((Namespace.PREFIX_FILE.value, Namespace.PREFIX_IMAGE.value)):
        return None
    if '|' in text:  # deal with invalid markup in wikilinks
        text = text[text.rindex('|') + 1:].strip()
    return text


def get_resource_name_for_wikilink(wikilink: wtp.WikiLink) -> Optional[str]:
    return label2name(str_util.capitalize(_remove_language_tag(wikilink.target.strip()))) if wikilink.target else None


def get_resource_idx_for_resource_name(res_name: str) -> Optional[int]:
    dbr = DbpResourceStore.instance()
    if dbr.has_resource_with_name(res_name):
        res = dbr.get_resource_by_name(res_name)
        if not res.is_meta:
            return res.idx
        redirected_res = dbr.resolve_spelling_redirect(res)
        if not redirected_res.is_meta:
            return redirected_res.idx
    return None


def _remove_language_tag(link_target: str) -> str:
    if not link_target or link_target[0] != ':':
        return link_target
    if len(link_target) < 4 or link_target[3] != ':':
        return link_target[1:]
    return link_target[4:]
