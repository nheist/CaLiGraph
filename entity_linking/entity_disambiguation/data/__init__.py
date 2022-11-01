from typing import Tuple
import utils
from .util import CorpusType, DataCorpus, Pair
from .listing import ListingDataCorpus, _init_listing_data_corpora
from .nilk import NilkDataCorpus, _init_nilk_data_corpora


def get_data_corpora(corpus_type: CorpusType) -> Tuple[DataCorpus, DataCorpus, DataCorpus]:
    if corpus_type == CorpusType.LIST:
        return utils.load_or_create_cache('ED_datasets', _init_listing_data_corpora)
    elif corpus_type == CorpusType.NILK:
        return utils.load_or_create_cache('ED_nilk_datasets', _init_nilk_data_corpora)
    raise ValueError(f'Unknown corpus type: {corpus_type.name}')