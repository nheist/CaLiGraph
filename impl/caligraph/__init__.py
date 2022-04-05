"""Functionality to retrieve cached versions of caligraph in several stages."""

from impl.caligraph.graph import CaLiGraph
from impl.caligraph.entity import ClgEntityStore
import impl.caligraph.serialize as clg_serialize
from impl import listing
from . import cali2ax


def extract_and_serialize():
    """Extract and serialize CaLiGraph."""
    clge = ClgEntityStore.instance()
    clge.add_axiom_information(cali2ax.get_axiom_information())
    clge.add_listing_information(listing.get_entity_information_from_listings())
    clg_serialize.run_serialization()
