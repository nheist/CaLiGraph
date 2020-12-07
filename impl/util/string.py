def capitalize(text: str) -> str:
    if not text:
        return text
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:]


def regularize_spaces(text: str) -> str:
    return ''.join(text.split())
