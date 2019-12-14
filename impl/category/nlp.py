import impl.util.nlp as nlp_util
import impl.category.store as cat_store
from spacy.tokens import Doc


def parse_category(category: str) -> Doc:
    """Return the category name as parsed Doc."""
    label = cat_store.get_label(category)
    return nlp_util.parse(label)
