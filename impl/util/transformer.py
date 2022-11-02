from enum import Enum
from impl.util.nlp import EntityTypeLabel


class EntityIndex(Enum):
    NEW_ENTITY = -1
    NO_ENTITY = -2
    IGNORE = -100


class SpecialToken(Enum):
    # CONTEXT TOKENS
    CONTEXT_SEP = '[CXS]'
    CONTEXT_END = '[CXE]'
    TEXT_SEP = '[TXS]'
    TABLE_ROW = '[ROW]'
    TABLE_COL = '[COL]'
    ENTRY_L1 = '[E1]'
    ENTRY_L2 = '[E2]'
    ENTRY_L3 = '[E3]'

    # TYPE TOKENS
    PERSON = '[TPE]'
    NORP = '[TNO]'
    FAC = '[TFA]'
    ORG = '[TOR]'
    GPE = '[TGP]'
    LOC = '[TLO]'
    PRODUCT = '[TPR]'
    EVENT = '[TEV]'
    WORK_OF_ART = '[TWO]'
    LAW = '[TLA]'
    LANGUAGE = '[TLN]'
    SPECIES = '[TSP]'
    OTHER = '[TOT]'

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

    @classmethod
    def get_type_token(cls, type_label: EntityTypeLabel) -> str:
        return cls[type_label.name].value
