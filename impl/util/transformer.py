from enum import Enum


class EntityIndex(Enum):
    NEW_ENTITY = -1
    NO_ENTITY = -2
    IGNORE = -100


class SpecialToken(Enum):
    CONTEXT_SEP = '[CXS]'
    CONTEXT_END = '[CXE]'
    TABLE_ROW = '[ROW]'
    TABLE_COL = '[COL]'
    ENTRY_L1 = '[E1]'
    ENTRY_L2 = '[E2]'
    ENTRY_L3 = '[E3]'

    @classmethod
    def all_tokens(cls):
        return {t.value for t in cls}

    @classmethod
    def item_starttokens(cls):
        return {cls.TABLE_ROW.value, cls.ENTRY_L1.value, cls.ENTRY_L2.value, cls.ENTRY_L3.value}

    @classmethod
    def get_entry_by_depth(cls, depth: int):
        if depth == 1:
            return cls.ENTRY_L1.value
        elif depth == 2:
            return cls.ENTRY_L2.value
        elif depth >= 3:
            return cls.ENTRY_L3.value
        raise ValueError(f'Trying to retrieve a transformer special token for an entry of depth {depth}.')
