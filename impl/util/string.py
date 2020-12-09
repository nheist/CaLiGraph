def capitalize(text: str) -> str:
    if not text:
        return text
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:]


def regularize_spaces(text: str) -> str:
    return ' '.join(text.split())


def transfer_word_casing(source_word: str, target_word: str) -> str:
    new_target_word = []
    for idx, c in enumerate(target_word):
        if idx >= len(source_word):
            new_target_word.append(c)
        elif source_word[idx].isupper():
            new_target_word.append(c.upper())
        else:
            new_target_word.append(c.lower())
    return ''.join(new_target_word)
