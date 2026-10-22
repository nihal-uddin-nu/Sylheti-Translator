from pathlib import Path
import pandas as pd
import json
import re


# ==================================================
# PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

EN_BN_TXT = RAW_DATA_DIR / "English-Bangla.txt"
SYL_BN_CSV = RAW_DATA_DIR / "Sylheti-Bangla.csv"

RESOURCES_DIR = ROOT_DIR / "resources"
AUTOCOMPLETE_DIR = RESOURCES_DIR / "autocomplete"

WORD_TRIE_DIR = AUTOCOMPLETE_DIR / "word_tries"
SENTENCE_TRIE_DIR = AUTOCOMPLETE_DIR / "sentence_tries"

SYLHETI_WORD_TRIE = WORD_TRIE_DIR / "sylheti_word_trie.json"
BANGLA_WORD_TRIE = WORD_TRIE_DIR / "bangla_word_trie.json"
ENGLISH_WORD_TRIE = WORD_TRIE_DIR / "english_word_trie.json"

SYLHETI_SENTENCE_TRIE = SENTENCE_TRIE_DIR / "sylheti_sentence_trie.json"
BANGLA_SENTENCE_TRIE = SENTENCE_TRIE_DIR / "bangla_sentence_trie.json"
ENGLISH_SENTENCE_TRIE = SENTENCE_TRIE_DIR / "english_sentence_trie.json"


# ==================================================
# NORMALIZATION
# ==================================================

def normalize_sylheti(text: str) -> str:
    return text.lower().strip()


def normalize_bangla(text: str) -> str:
    return text.strip()


def normalize_english(text: str) -> str:
    return text.strip().lower()


# ==================================================
# TRIE CREATION HELPER FUNCTIONS
# ==================================================

def _build_word_trie(sentence: set[str], existing_trie=None) -> dict:
    """Builds a word trie from a set of words."""
    trie = {}
    if existing_trie is not None:
        trie = existing_trie.copy()

    for word in sentence:
        node = trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node["end"] = True

    return trie


def _build_sentence_trie(sentence: list[str], existing_trie=None) -> dict:
    """Builds a sentence trie from a list of words."""
    trie = {}
    if existing_trie is not None:
        trie = existing_trie.copy()

    node = trie
    for word in sentence:
        if word not in node:
            node[word] = {}
        node = node[word]
    node["end"] = True

    return trie


# ==================================================
# BUILD WORD TRIES
# ==================================================

def build_sylheti_word_trie(output=SYLHETI_WORD_TRIE) -> None:
    data = set()
    
    # Read the Sylheti-Bangla CSV file and extract Sylheti words
    df = pd.read_csv(SYL_BN_CSV)
    df = df[["sylheti"]].drop_duplicates()

    sentences = df["sylheti"].tolist()
    for sentence in sentences:
        words = re.findall(r'\w+', normalize_sylheti(sentence))
        data.update(words)

    # Build the word trie for Sylheti words
    trie = _build_word_trie(data)

    # Save the word trie to a JSON file
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(trie, f, ensure_ascii=False, indent=4)

    print(f"\nSaved Sylheti word trie to:")
    print(f"  {output}")

def build_bangla_word_trie(output=BANGLA_WORD_TRIE) -> None:
    data = set()
    
    # Read the Sylheti-Bangla CSV file and extract Bangla words
    df = pd.read_csv(SYL_BN_CSV)
    df = df[["bangla"]].drop_duplicates()

    sentences = df["bangla"].tolist()
    for sentence in sentences:
        words = re.findall(r'\w+', normalize_bangla(sentence))
        data.update(words)

    # Read the English-Bangla text file and extract Bangla words
    text = EN_BN_TXT

    with open(text, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
        
                    if not line:
                        continue
        
                    parts = line.split("\t")
        
                    if len(parts) < 3:
                        continue
        
                    bn = normalize_english(parts[1])
                    words = re.findall(r'\w+', bn)
                    data.update(words)

    # Build the word trie for Bangla words
    trie = _build_word_trie(data)

    # Save the word trie to a JSON file
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(trie, f, ensure_ascii=False, indent=4)

    print(f"\nSaved Bangla word trie to:")
    print(f"  {output}")


def build_english_word_trie(output=ENGLISH_WORD_TRIE) -> None:
    data = set()
    
    # Read the English-Bangla text file and extract English words
    text = EN_BN_TXT
    
    with open(text, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
    
                if not line:
                    continue
    
                parts = line.split("\t")
    
                if len(parts) < 3:
                    continue
    
                en = normalize_english(parts[0])
                words = re.findall(r'\w+', en)
                data.update(words)

    # Build the word trie for English words
    trie = _build_word_trie(data)

    # Save the word trie to a JSON file
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(trie, f, ensure_ascii=False, indent=4)

    print(f"\nSaved English word trie to:")
    print(f"  {output}")


def build_word_tries() -> None:
    build_sylheti_word_trie()
    build_bangla_word_trie()
    build_english_word_trie()


# ==================================================
# BUILD SENTENCE TRIES
# ==================================================

def build_sylheti_sentence_trie(output=SYLHETI_SENTENCE_TRIE) -> None:
    trie = {}
    
    # Read the Sylheti-Bangla CSV file and extract Sylheti words
    df = pd.read_csv(SYL_BN_CSV)
    df = df[["sylheti"]].drop_duplicates()

    sentences = df["sylheti"].tolist()
    for sentence in sentences:
        words = re.findall(r'\w+', normalize_sylheti(sentence))

        # Add to the sentence trie for Sylheti words
        trie = _build_sentence_trie(words, existing_trie=trie)
        
    # Save the sentence trie to a JSON file
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(trie, f, ensure_ascii=False, indent=4)

    print(f"\nSaved Sylheti sentence trie to:")
    print(f"  {output}")

def build_bangla_sentence_trie(output=BANGLA_SENTENCE_TRIE) -> None:
    trie = {}
    
    # Read the Sylheti-Bangla CSV file and extract Bangla sentences
    df = pd.read_csv(SYL_BN_CSV)
    df = df[["bangla"]].drop_duplicates()

    sentences = df["bangla"].tolist()
    for sentence in sentences:
        words = re.findall(r'\w+', normalize_bangla(sentence))

        # Add to the sentence trie for Bangla words
        trie = _build_sentence_trie(words, existing_trie=trie)

    # Read the English-Bangla text file and extract Bangla sentences
    text = EN_BN_TXT
        
    with open(text, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
    
                if not line:
                    continue
    
                parts = line.split("\t")
    
                if len(parts) < 3:
                    continue
    
                bn = normalize_bangla(parts[1])
                words = re.findall(r'\w+', bn)

                trie = _build_sentence_trie(words, existing_trie=trie)
        
    # Save the sentence trie to a JSON file
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(trie, f, ensure_ascii=False, indent=4)

    print(f"\nSaved Bangla sentence trie to:")
    print(f"  {output}")

def build_english_sentence_trie(output=ENGLISH_SENTENCE_TRIE) -> None:
    trie = {}

    # Read the English-Bangla text file and extract English sentences
    text = EN_BN_TXT

    with open(text, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 3:
                continue

            en = normalize_english(parts[0])
            words = re.findall(r'\w+', en)

            trie = _build_sentence_trie(words, existing_trie=trie)

    # Save the sentence trie to a JSON file
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(trie, f, ensure_ascii=False, indent=4)

    print(f"\nSaved English sentence trie to:")
    print(f"  {output}")


def build_sentence_tries() -> None:
    build_sylheti_sentence_trie()
    build_bangla_sentence_trie()
    build_english_sentence_trie()