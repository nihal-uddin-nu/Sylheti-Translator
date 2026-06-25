def detect_script(text):
    has_bangla = False
    has_english = False

    for ch in text:
        if '\u0980' <= ch <= '\u09FF':
            has_bangla = True
        elif ('a' <= ch.lower() <= 'z'):
            has_english = True

    if has_bangla and not has_english:
        return "Bangla"
    elif has_english and not has_bangla:
        return "English"
    elif has_bangla and has_english:
        return "Mixed"
    else:
        return "Unknown"