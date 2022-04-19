"""Combine extracted SE labels with information in page markup to get extracted SE entities with labels."""

from typing import Dict, Tuple
from impl.dbpedia.resource import DbpResource, DbpEntity, DbpResourceStore
from impl import wikipedia
from collections import defaultdict
import impl.wikipedia.wikimarkup_parser as wmp


def match_entities_with_uris(subject_entities_per_page: Dict[DbpResource, dict]) -> Dict[DbpResource, dict]:
    parsed_pages = wikipedia.get_parsed_pages()
    return {res: _match_entities_for_page(res, entities_per_ts, parsed_pages[res]) for res, entities_per_ts in subject_entities_per_page.items()}


def _match_entities_for_page(page_res: DbpResource, entities_per_ts: dict, page_content: dict) -> dict:
    dbr = DbpResourceStore.instance()
    
    enriched_entities = defaultdict(lambda: defaultdict(dict))
    page_entity_map, section_entity_map = _create_entity_maps(page_content)
    for ts, entities_per_s in entities_per_ts.items():
        for s, entities in entities_per_s.items():
            for ent_text, ent_tag in entities.items():
                if ent_text in page_entity_map[ts][s]:
                    ent_idx = page_entity_map[ts][s][ent_text]
                    if not isinstance(dbr.get_resource_by_idx(ent_idx), DbpEntity):
                        continue  # discard anything that is not an entity (listpage, file, ..)
                    ent_name = dbr.get_resource_by_idx(ent_idx)
                else:
                    section_part = f'#{s}' if s != 'Main' else ''
                    ent_name = f'{page_res.name}{section_part}--{ent_text}'
                enriched_entities[ts][s][ent_name] = {
                    'text': ent_text,
                    'tag': ent_tag,
                    'TS_entidx': section_entity_map[ts],
                    'S_entidx': section_entity_map[s]
                }
    return {ts: dict(enriched_entities[ts]) for ts in enriched_entities}


def _create_entity_maps(page_content: dict) -> Tuple[dict, dict]:
    page_entity_map = defaultdict(lambda: defaultdict(dict))
    section_entity_map = defaultdict(lambda: None)
    top_section_name = ''
    for section_data in page_content['sections']:
        section_name_markup = section_data['name']
        section_name = wmp.wikitext_to_plaintext(section_name_markup)
        section_entity_map[section_name] = wmp.get_first_wikilink_resource(section_name_markup)
        top_section_name = section_name if section_data['level'] <= 2 else top_section_name
        for enum_data in section_data['enums']:
            for entry in enum_data:
                for entity in entry['entities']:
                    page_entity_map[top_section_name][section_name][entity['text']] = entity['idx']
        for table in section_data['tables']:
            for row in table['data']:
                for cell in row:
                    for entity in cell['entities']:
                        page_entity_map[top_section_name][section_name][entity['text']] = entity['idx']
    return page_entity_map, section_entity_map
