"""Combine extracted SE labels with information in page markup to get extracted SE entities with labels."""


from impl import wikipedia
from collections import defaultdict
import impl.dbpedia.util as dbp_util
import impl.wikipedia.wikimarkup_parser as wmp


def match_entities_with_uris(subject_entities_per_page: dict) -> dict:
    enriched_entities_per_page = {}

    parsed_pages = wikipedia.get_parsed_articles()
    for page_uri, entities_per_ts in subject_entities_per_page.items():
        page_name = dbp_util.resource2name(page_uri)
        enriched_entities = defaultdict(lambda: defaultdict(dict))
        page_entity_map, section_entity_map = _create_entity_maps(parsed_pages[page_uri])
        for ts, entities_per_s in entities_per_ts.items():
            for s, entities in entities_per_s.items():
                for ent_text, ent_tag in entities.items():
                    if ent_text in page_entity_map[ts][s]:
                        ent_name = page_entity_map[ts][s][ent_text]
                    else:
                        ent_name = f'{page_name}--{ent_text}'
                    enriched_entities[ts][s][ent_name] = {
                        'text': ent_text,
                        'tag': ent_tag,
                        'TS_ent': section_entity_map[ts],
                        'S_ent': section_entity_map[s]
                    }

        enriched_entities_per_page[page_uri] = {ts: dict(enriched_entities[ts]) for ts in enriched_entities}

    return enriched_entities_per_page


def _create_entity_maps(page_markup: dict) -> tuple:
    page_entity_map = defaultdict(lambda: defaultdict(dict))
    section_entity_map = defaultdict(lambda: None)
    top_section_name = ''
    for section_data in page_markup['sections']:
        section_name_markup = section_data['name']
        section_name = wmp.wikitext_to_plaintext(section_name_markup)
        section_entity_map[section_name] = wmp.get_first_wikilink_entity(section_name_markup)
        top_section_name = section_name if section_data['level'] <= 2 else top_section_name

        for enum_data in section_data['enums']:
            for entry in enum_data:
                for entity in entry['entities']:
                    page_entity_map[top_section_name][section_name][entity['text']] = entity['name']
        for table in section_data['tables']:
            for row in table['data']:
                for cell in row:
                    for entity in cell['entities']:
                        page_entity_map[top_section_name][section_name][entity['text']] = entity['name']
    return page_entity_map, section_entity_map
