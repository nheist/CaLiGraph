from spacy.tokens import Doc, Span
import impl.util.words as words_util


# TAGGING OF LEXICAL HEAD OF A CATEGORY OR LIST PAGE


def tag_lexical_head(doc: Doc) -> Doc:
    """Tag the lexical head of a set with the entity tag 'LH'."""
    if len(doc) == 0:
        return doc

    # ensure that numbers are also regarded as nouns if being stand-alone
    if doc[0].tag_ == 'CD' and (len(doc) < 2 or not doc[1].tag_.startswith('NN')):
        doc.ents = [Span(doc, 0, 1, label=doc.vocab.strings['LH'])]
        return doc

    chunk_words = {w for chunk in doc.noun_chunks for w in chunk}
    lexhead_start = None
    for chunk in doc.noun_chunks:
        # find the lexical head by looking for plural nouns (and ignore things like parentheses, conjunctions, ..)
        elem = chunk.root
        if elem.tag_ == 'NNP' and words_util.is_english_plural_word(elem.text):
            # fix plural nouns that are parsed incorrectly as proper nouns due to capitalization in the beginning
            elem.tag = doc.vocab.strings['NNS']
        if elem.tag_ not in ['NN', 'NNS']:
            continue
        if len(doc) > elem.i + 1:
            if doc[elem.i+1].text[0] in ["'", "´", "`"]:
                continue
            if doc[elem.i+1].text in ['(', ')', '–'] and doc[-1].text != ')':
                continue
            if doc[elem.i + 1].tag_ in ['NN', 'NNS'] or (len(doc) > elem.i + 2 and doc[elem.i+1].text in ['and', 'or', ','] and doc[elem.i+2] in chunk_words):
                lexhead_start = lexhead_start if lexhead_start is not None else chunk.start
                continue
        lexhead_start = lexhead_start if lexhead_start is not None else chunk.start
        doc.ents = [Span(doc, i, i+1, label=doc.vocab.strings['LH']) for i in range(lexhead_start, chunk.end)]
        break
    return doc


def tag_lexical_head_subjects(doc: Doc) -> Doc:
    """Tag the main subjects of lexical heads with 'LHS' - but only if they are in plural."""
    lh_entities = list(doc.ents)
    lhs_entities = []

    subject_connectors = ['and', 'or', ',']
    # iterate over the lexical head of the doc in reversed order and mark the plural nouns as subjects
    for w in [w for w in reversed(doc) if w.ent_type_ == 'LH']:
        if w.text in subject_connectors:
            continue  # ignore subject connectors
        if w.tag_ != 'NNS':
            break  # LHS are at the end of the LH, so we stop if we are not looking at a plural noun
        # add current word to LHS
        lhs_entities.append(Span(doc, w.i, w.i+1, label=doc.vocab.strings['LHS']))
        lh_entities = [e for e in lh_entities if e.start != w.i]
        if w.i == 0 or doc[w.i-1].text not in subject_connectors:
            break  # if the previous word is not a subject connector we found all LHS
    doc.ents = lh_entities + lhs_entities
    return doc


# TAGGING OF THE BY-PHRASE OF A CATEGORY OR LIST PAGE


BY_PHRASE_EXCEPTIONS = {'bell hooks', 'DBC Pierre'}


def tag_by_phrase(doc: Doc) -> Doc:
    """Tag the 'by'-phrase at the end of a taxonomic set with 'BY', e.g. 'People by country' -> tag 'by country'"""
    # locate all by-phrases
    by_indices = [w.i for w in doc if w.text == 'by']
    if len(by_indices) == 0:
        return doc
    end_index = len(doc)
    if ' in ' in doc[by_indices[-1]:].text:
        # ignore information after the by phrase (e.g. do not tag 'in Honduras' for 'People by city in Honduras')
        end_index = [w.i for w in doc[by_indices[-1]:] if w.text == 'in'][0]
    # find valid by-phrases
    for idx, by_index in enumerate(by_indices):
        if by_index == 0 or by_index == len(doc) - 1:
            continue
        current_doc = doc[:end_index] if len(by_indices) == idx+1 else doc[:by_indices[idx+1]]
        text_after_by = current_doc[by_index+1:].text.strip()
        word_before_by = current_doc[by_index - 1]
        word_after_by = current_doc[by_index + 1]
        if text_after_by in BY_PHRASE_EXCEPTIONS:
            continue
        if word_after_by.text[0].isupper() and (word_after_by.text.endswith('.') or not word_after_by.text.isupper()):
            continue
        if any(w.tag_ == 'NNS' for w in doc[by_index+1:]):
            continue
        if word_before_by.tag_ == 'VBN':
            continue
        if word_after_by.text in ['a', 'an', 'the']:
            continue
        if doc[by_index - 1].text == '(':  # include possible parenthesis before by phrase
            by_index -= 1
        doc.ents = [Span(doc, by_index, end_index, label=doc.vocab.strings['BY'])]
    return doc
