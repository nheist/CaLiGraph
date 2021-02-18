import wikitextparser as wtp
from typing import Union, Optional
import re
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.util.string as str_util


def wikitext_to_plaintext(text: Union[str, wtp.WikiText]) -> str:
    try:
        parsed_text = wtp.parse(text) if type(text) == str else text
        # bolds and italics are already removed during preprocessing to reduce runtime
        result = parsed_text.plain_text(replace_bolds_and_italics=False).strip(" '\t\n")
        result = re.sub(r'\n+', '\n', result)
        result = re.sub(r' +', ' ', result)
        return result
    except (AttributeError, IndexError):
        return text


def get_first_wikilink_entity(text: Union[str, wtp.WikiText]) -> Optional[str]:
    try:
        parsed_text = wtp.parse(text) if type(text) == str else text
        for wl in parsed_text.wikilinks:
            return get_entity_for_wikilink(wl)
        return None
    except (AttributeError, IndexError):
        return None


def get_entity_for_wikilink(wikilink: wtp.WikiLink) -> Optional[str]:
    if not wikilink.target:
        return None
    link_target = _remove_language_tag(wikilink.target.strip())
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
