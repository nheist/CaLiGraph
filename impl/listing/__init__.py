"""Extract entities from listings in arbitrary Wikipedia pages.
Learn association rules to gather types and relations for the extracted entities."""

from typing import Dict, Tuple
import utils
from . import extract


def get_entity_information_from_listings() -> Dict[int, Dict[Tuple[int, str], dict]]:
    listing_entity_information = utils.load_or_create_cache('listing_page_entities', extract.extract_listing_entity_information)
    return listing_entity_information
