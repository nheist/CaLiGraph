import utils
import os
import pickle


def store_candidates(approach_id: str, candidates: dict):
    with open(_get_approach_path(approach_id), mode='wb') as f:
        return pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_candidates(approach_id: str) -> dict:
    with open(_get_approach_path(approach_id), mode='rb') as f:
        return pickle.load(f)


def _get_approach_path(approach_id: str) -> str:
    return os.path.join(utils._get_root_path(), 'entity_linking', 'data', approach_id)
