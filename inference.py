import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = "./umt5_sylheti_model_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)

# -----------------------------
# Sylheti → Bangla
# -----------------------------
def translate_sylheti_to_bangla(text: str) -> str:
    # add same prefix used in training
    input_text = "translate Sylheti to Bangla: " + text
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Sylheti → English (via Bangla)
# -----------------------------
import argostranslate.package, argostranslate.translate

# Load installed packages if not already
# argostranslate.package.install_from_path("argostranslate_bangla_to_english.argosmodel")

def sylheti_to_english(text: str) -> str:
    bangla = translate_sylheti_to_bangla(text)
    return argostranslate.translate.translate(bangla, "bn", "en")

# -----------------------------
# Test
# -----------------------------
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="ami zaimu", help="Sylheti text to translate")
    args = parser.parse_args()

    sample = args.text
    print("Sylheti → Bangla:", translate_sylheti_to_bangla(sample))
    print("Sylheti → English:", sylheti_to_english(sample))