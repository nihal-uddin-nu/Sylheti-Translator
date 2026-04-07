import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_CSV = "data/train.csv"
MODEL_NAME = "google/umt5-small"
OUTPUT_DIR = "./umt5_sylheti_model_sanity"
MAX_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 2
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD DATA (small sample)
# -----------------------------
df = pd.read_csv(TRAIN_CSV).sample(500)
df = df.dropna()
dataset = Dataset.from_pandas(df)

# -----------------------------
# TOKENIZER & MODEL
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess(batch):
    inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    targets = tokenizer(
        batch["output"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in targets["input_ids"]
    ]
    inputs["labels"] = labels
    return inputs

dataset = dataset.map(preprocess, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------------
# TRAINING ARGUMENTS
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
)

# -----------------------------
# TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# -----------------------------
# TRAIN
# -----------------------------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    # -----------------------------
    # Quick inference check
    # -----------------------------
    test_samples = ["ami zaimu", "tumi koi zaiba", "amra bikel khai"]
    for s in test_samples:
        inputs = tokenizer(s, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_length=64)
        print(f"Sylheti: {s}")
        print(f"Bangla: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")