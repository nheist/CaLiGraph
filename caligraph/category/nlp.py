import caligraph.util.nlp as nlp_util
import caligraph.category.store as cat_store
import util


def is_conceptual(category: str) -> bool:
    label = cat_store.get_label(category)
    parsed_label = nlp_util.parse(label)
    util.get_logger().debug('{} ---> {}'.format(label, list(parsed_label.noun_chunks)))
    return False


def singularize(category: str) -> str:
    pass
