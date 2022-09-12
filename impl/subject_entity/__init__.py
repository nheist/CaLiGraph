from impl.subject_entity import mention_detection, entity_disambiguation


def extract_subject_entities():
    """Find mentions of subject entities and disambiguate them."""
    mention_detection.detect_mentions()
    entity_disambiguation.disambiguate_subject_entities()
