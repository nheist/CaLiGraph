"""Functionality to retrieve cached versions of caligraph in several stages."""

from impl.caligraph.graph import CaLiGraph
from impl.caligraph.ontology import ClgOntologyStore
from impl.caligraph.entity import ClgEntityStore
import impl.caligraph.serialize as clg_serialize
from impl.wikipedia import WikiPageStore
from impl import subject_entity, listing
from impl.caligraph import cali2ax


def extract_and_serialize():
    """Extract and serialize CaLiGraph."""
    # initialize ontology
    ClgOntologyStore.instance()
    # initialize parsed Wikipedia page corpus
    WikiPageStore.instance()
    # initalize entities from DBpedia
    clge = ClgEntityStore.instance()
    # run subject entity extraction
    subject_entity.extract_subject_entities()
    clge.add_subject_entities()
    # add information from CaLi2Ax axioms
    axiom_information = cali2ax.get_axiom_information()
    clge.add_axiom_information(axiom_information)
    # add information from listing axioms
    listing_information = listing.get_entity_information_from_listings()
    clge.add_listing_information(listing_information)
    # serialize final graph
    clg_serialize.run_serialization()
