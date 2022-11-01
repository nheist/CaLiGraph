from typing import Set, Tuple, Dict

from impl.caligraph.entity import ClgEntity
from impl.wikipedia import MentionId
from .util import DataCorpus


class NilkDataCorpus(DataCorpus):
    def get_mention_labels(self) -> Dict[MentionId, str]:
        pass  # TODO

    def get_mention_input(self, add_page_context: bool, add_text_context: bool) -> Tuple[Dict[MentionId, str], Dict[MentionId, bool]]:
        pass  # TODO

    def get_entities(self) -> Set[ClgEntity]:
        pass  # TODO


def _init_nilk_data_corpora() -> Tuple[NilkDataCorpus, NilkDataCorpus, NilkDataCorpus]:
    pass  # TODO
