from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# ==================================================
# PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

TRANSLATE_CSV = ROOT_DIR / "data" / "processed" / "translate.csv"
OUTPUT_DIR = ROOT_DIR / "umt5_sylheti_model"


# ==================================================
# TRAINING CONFIG
# ==================================================

MODEL_NAME = "google/umt5-small"

MAX_LENGTH = 128
BATCH_SIZE = 16

EPOCHS = 15
LEARNING_RATE = 3e-4

SEED = 42


# ==================================================
# PREPROCESSING
# ==================================================

def preprocess(batch, tokenizer):
    inputs = tokenizer(
        batch["input"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

    targets = tokenizer(
        batch["output"],
        truncation=True,
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
    df = pd.read_csv(TRANSLATE_CSV)
    df = df.dropna()

    return Dataset.from_pandas(df)


# ==================================================
# TRAINING
# ==================================================

def build_trainer() -> tuple[Trainer, AutoModelForSeq2SeqLM]:
    dataset = load_dataset()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
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
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        seed=SEED,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer, model


# ==================================================
# MAIN
# ==================================================

def main() -> None:
    trainer, _ = build_trainer()

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(str(OUTPUT_DIR))

    print(f"\nModel saved to:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()