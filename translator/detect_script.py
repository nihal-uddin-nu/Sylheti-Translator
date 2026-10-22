def detect_script(text):
    has_bengali = False
    has_latin = False

    for ch in text:
        if '\u0980' <= ch <= '\u09FF':
            has_bengali = True

        elif ('a' <= ch.lower() <= 'z'):
            has_latin = True

    match (has_bengali, has_latin):
        case (True, False):
            return "Bangla"
        
        case (False, True):
            return "Latin"
        
        case (True, True):
            return "Mixed"
        
        case _:
            return "Unknown"