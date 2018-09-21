import caligraph.util.nlp as nlp_util
import caligraph.category.store as cat_store


def is_conceptual(category: str) -> bool:
    label = cat_store.get_label(category)
    label_chunks = list(nlp_util.parse(label).noun_chunks)

    if len(label_chunks) == 0:
        return False

    return label_chunks[0][-1].tag_ == 'NNS'


def singularize(category: str) -> str:
    pass
