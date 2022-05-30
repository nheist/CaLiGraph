from typing import Set, List, Tuple
from collections import defaultdict
from unidecode import unidecode
from nltk.corpus import stopwords
import re
from impl.dbpedia.resource import DbpResourceStore


class WordBlocker:
    def __init__(self):
        self.entity_blocks = defaultdict(set)
        for res in DbpResourceStore.instance().get_embedding_vectors():
            for sf in res.get_surface_forms():
                self.entity_blocks[self._block_surface_form(sf.split())].add(res.idx)

    def get_entity_indices_for_words(self, words: List[str]) -> Set[int]:
        return self.entity_blocks[self._block_surface_form(words)]

    def _block_surface_form(self, sf_words: List[str]) -> Tuple[str]:
        sf_words = {unidecode(w.lower()) for w in sf_words}  # lowercase everything and convert special chars
        sf_words = {self._make_alphanum(w) for w in sf_words if w}  # remove non-alphanumeric characters
        filtered_sf_words = sf_words.difference(stopwords.words('english'))
        return tuple(sorted(filtered_sf_words or sf_words))

    @classmethod
    def _make_alphanum(cls, text: str) -> str:
        text_alphanum = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        return text_alphanum if len(text_alphanum) > 2 else text
