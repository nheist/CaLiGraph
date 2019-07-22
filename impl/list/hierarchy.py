from . import util as list_util
import impl.category.base as cat_base
import impl.category.store as cat_store
import impl.category.util as cat_util
import impl.category.wikitaxonomy as cat_wikitax
from impl.category.conceptual import is_conceptual_category
import impl.dbpedia.store as dbp_store
import impl.list.store as list_store
import util
import impl.util.nlp as nlp_util
from collections import defaultdict
from spacy.tokens import Doc


"""Hierarchy of listcategories to be incorporated in the category-lists-hierarchy

Listcategories are incorporated in one of three ways:
- Equivalence Mapping: The listcategory is mapped directly to a category (no new hierarchy item is created)
- Child Mapping: The listcategory is added to the hierarchy as the child of an existing hierarchy item (category or listcategory)
- Root Mapping: A special case of "Child Mapping" where no parent can be found and the listcategory is added as child of the root hierarchy item
"""


def get_hierarchy_mappings():
    pass
