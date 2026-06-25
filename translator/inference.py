from pathlib import Path
import argparse

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# ==================================================
# PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "umt5_sylheti_model"


# ==================================================
# MODEL CONFIG
# ==================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 128
NUM_BEAMS = 4


# ==================================================
# MODEL LOADING
# ==================================================

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(MODEL_DIR)
        .to(DEVICE)
    )

    return tokenizer, model


tokenizer, model = load_model()


# ==================================================
# TRANSLATION
# ==================================================

def generate_translation(
    prompt: str,
    max_length: int = MAX_LENGTH,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=NUM_BEAMS,
        early_stopping=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        length_penalty=1.0,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


# ==================================================
# LANGUAGE PAIRS
# ==================================================

def translate_sylheti_to_bangla(text: str) -> str:
    return generate_translation(
        f"Translate Sylheti to Bangla: {text}"
    )


def translate_bangla_to_sylheti(text: str) -> str:
    return generate_translation(
        f"Translate Bangla to Sylheti: {text}"
    )


def translate_bangla_to_english(text: str) -> str:
    return generate_translation(
        f"Translate Bangla to English: {text}"
    )


def translate_sylheti_to_english(text: str) -> str:
    bangla = translate_sylheti_to_bangla(text)

    return translate_bangla_to_english(bangla)


# ==================================================
# CLI
# ==================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate Sylheti text."
    )

    parser.add_argument(
        "--text",
        type=str,
        default="আমি যাইমু",
        help="Sylheti text to translate",
    )

    return parser.parse_args()


# ==================================================
# MAIN
# ==================================================

def main() -> None:
    args = parse_args()

    print(f"Input              : {args.text}")
    print(
        f"Sylheti → Bangla   : "
        f"{translate_sylheti_to_bangla(args.text)}"
    )
    print(
        f"Sylheti → English  : "
        f"{translate_sylheti_to_english(args.text)}"
    )


if __name__ == "__main__":
    main()