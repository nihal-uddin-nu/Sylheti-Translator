from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ==================================================
# PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = ROOT_DIR / "data" / "processed" / "train.csv"
OUTPUT_DIR = ROOT_DIR / "umt5_sylheti_model_sanity"


# ==================================================
# TRAINING CONFIG
# ==================================================

MODEL_NAME = "google/umt5-small"

MAX_LENGTH = 64
BATCH_SIZE = 8

EPOCHS = 2
LEARNING_RATE = 5e-5

SAMPLE_SIZE = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# PREPROCESSING
# ==================================================

def preprocess(batch, tokenizer):
    inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    targets = tokenizer(
        batch["output"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    labels = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in sequence
        ]
        for sequence in targets["input_ids"]
    ]

    inputs["labels"] = labels

    return inputs


# ==================================================
# DATA LOADING
# ==================================================

def load_dataset() -> Dataset:
    df = (
        pd.read_csv(TRAIN_CSV)
        .dropna()
        .sample(SAMPLE_SIZE, random_state=42)
    )

    return Dataset.from_pandas(df)


# ==================================================
# TRAINER SETUP
# ==================================================

def build_trainer():
    dataset = load_dataset()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(MODEL_NAME)
        .to(DEVICE)
    )

    dataset = dataset.map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
    )

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels",
        ],
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    return trainer, tokenizer, model


# ==================================================
# INFERENCE
# ==================================================

def generate_translation(
    text: str,
    tokenizer,
    model,
) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_length=MAX_LENGTH,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


# ==================================================
# MAIN
# ==================================================

def main() -> None:
    trainer, tokenizer, model = build_trainer()

    trainer.train()

    trainer.save_model(str(OUTPUT_DIR))

    print(f"\nModel saved to:")
    print(f"  {OUTPUT_DIR}")

    print("\nQuick inference check:\n")

    test_samples = [
        "ami zaimu",
        "tumi koi zaiba",
        "amra bikel khai",
    ]

    for sample in test_samples:
        translation = generate_translation(
            sample,
            tokenizer,
            model,
        )

        print(f"Sylheti : {sample}")
        print(f"Bangla  : {translation}")
        print()


if __name__ == "__main__":
    main()