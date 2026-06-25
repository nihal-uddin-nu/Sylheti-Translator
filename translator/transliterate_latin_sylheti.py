import re

# --- Consonants ---
CONSONANTS = {
    "kh": "খ", "gh": "ঘ", "chh": "ছ", "ch": "চ", "jh": "ঝ",
    "th": "থ", "dh": "ধ", "ph": "ফ", "bh": "ভ",

    "k": "ক", "g": "গ", "j": "জ",
    "t": "ত", "d": "দ", "n": "ন",
    "p": "প", "b": "ব", "m": "ম",
    "r": "র", "l": "ল",
    "sh": "শ", "s": "স", "h": "হ",
    "y": "য", "f": "ফ", "v": "ব"
}

# --- Independent vowels ---
INDEPENDENT_VOWELS = {
    "aa": "আ", "ii": "ঈ", "uu": "ঊ",
    "a": "অ", "i": "ই", "u": "উ",
    "e": "এ", "o": "ও",
    "oi": "ঐ", "ou": "ঔ"
}

# --- Vowel signs ---
VOWEL_SIGNS = {
    "aa": "া", "i": "ি", "ii": "ী",
    "u": "ু", "uu": "ূ",
    "e": "ে", "oi": "ৈ",
    "o": "ো", "ou": "ৌ"
}

# --- Common conjuncts ---
CONJUNCTS = {
    "kk": "ক্ক",
    "kt": "ক্ত",
    "nd": "ন্দ",
    "nt": "ন্ত",
    "ngk": "ঙ্ক",
    "ngg": "ঙ্গ",
    "mp": "ম্প",
    "mb": "ম্ব",
    "ndh": "ন্ধ",
    "nch": "ঞ্চ",
    "nj": "ঞ্জ",
    "tt": "ট্ট",
    "dd": "ড্ড",
    "tr": "ত্র",
    "dr": "দ্র",
    "pr": "প্র",
    "br": "ব্র",
    "kr": "ক্র",
    "gr": "গ্র",
}

# --- Nasal normalization ---
def handle_nasal(text):
    # ng → ং when at end or before space
    text = re.sub(r'ng\b', 'ং', text)
    return text


def transliterate(text):
    text = text.lower()
    text = handle_nasal(text)

    keys = sorted(
        list(CONJUNCTS.keys()) +
        list(CONSONANTS.keys()) +
        list(INDEPENDENT_VOWELS.keys()),
        key=len,
        reverse=True
    )

    i = 0
    output = ""
    prev_was_consonant = False

    while i < len(text):
        matched = False

        for k in keys:
            if text[i:i+len(k)] == k:

                # --- conjunct ---
                if k in CONJUNCTS:
                    output += CONJUNCTS[k]
                    prev_was_consonant = True

                # --- vowel ---
                elif k in INDEPENDENT_VOWELS:
                    if prev_was_consonant:
                        output += VOWEL_SIGNS.get(k, "")
                    else:
                        output += INDEPENDENT_VOWELS[k]
                    prev_was_consonant = False

                # --- consonant ---
                elif k in CONSONANTS:
                    output += CONSONANTS[k]
                    prev_was_consonant = True

                i += len(k)
                matched = True
                break

        if not matched:
            output += text[i]
            prev_was_consonant = False
            i += 1

    return output

#print(transliterate(input()))