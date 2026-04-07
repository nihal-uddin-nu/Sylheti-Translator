import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
RAW_CSV = "data/Syl-Ban.csv"      # original CSV
OUTPUT_CSV = "data/train.csv"     # processed bidirectional CSV

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def normalize_sylheti(text: str) -> str:
    """Basic normalization for Sylheti."""
    text = text.lower().strip()
    # Add any custom replacements as needed
    return text

def normalize_bangla(text: str) -> str:
    """Basic normalization for Bangla."""
    return text.strip()

# -----------------------------
# LOAD RAW DATA
# -----------------------------
df = pd.read_csv(RAW_CSV)
print(f"Loaded {len(df)} rows from {RAW_CSV}")

# -----------------------------
# CREATE BIDIRECTIONAL DATA
# -----------------------------
data = []
for _, row in df.iterrows():
    syl = normalize_sylheti(row["sylheti"])
    bn  = normalize_bangla(row["bangla"])

    # Sylheti → Bangla
    data.append({
        "input": f"translate Sylheti to Bangla: {syl}",
        "output": bn
    })
    # Bangla → Sylheti
    data.append({
        "input": f"translate Bangla to Sylheti: {bn}",
        "output": syl
    })

# -----------------------------
# SAVE TRAIN CSV
# -----------------------------
train_df = pd.DataFrame(data)
train_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(train_df)} rows to {OUTPUT_CSV}")