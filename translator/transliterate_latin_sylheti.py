from pathlib import Path
import json
import re


# ==================================================
# LOAD TRANSLITERATION RESOURCES
# ==================================================

RESOURCE_DIR = (
    Path(__file__).resolve().parent.parent
    / "resources"
    / "transliteration"
    / "latin_to_bengali_script"
)

with open(RESOURCE_DIR / "consonants.json", encoding="utf-8") as f:
    CONSONANTS = json.load(f)

with open(RESOURCE_DIR / "vowels.json", encoding="utf-8") as f:
    VOWELS = json.load(f)

with open(RESOURCE_DIR / "conjuncts.json", encoding="utf-8") as f:
    CONJUNCTS = json.load(f)


# ==================================================
# NASAL HANDLING
# ==================================================

def handle_nasal(text):
    # ng → ং when at end of a word
    return re.sub(r"ng\b", "ং", text)


# ==================================================
# TRANSLITERATION
# ==================================================

def transliterate(text):
    text = handle_nasal(text.lower())

    keys = sorted(
        list(CONJUNCTS.keys()) +
        list(CONSONANTS.keys()) +
        list(VOWELS.keys()),
        key=len,
        reverse=True,
    )

    i = 0
    output = ""
    prev_was_consonant = False

    while i < len(text):
        matched = False

        for k in keys:
            if text.startswith(k, i):

                # --- conjunct ---
                if k in CONJUNCTS:
                    output += CONJUNCTS[k]
                    prev_was_consonant = True

                # --- vowel ---
                elif k in VOWELS:
                    if prev_was_consonant:
                        output += VOWELS[k]["sign"]
                    else:
                        output += VOWELS[k]["independent"]
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


# ==================================================
# MAIN
# ==================================================

def main() -> None:
    print(transliterate(input()))


if __name__ == "__main__":
    main()