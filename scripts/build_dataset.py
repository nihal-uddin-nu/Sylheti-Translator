from pathlib import Path

import pandas as pd


# ==================================================
# PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

EN_BN_TXT = RAW_DATA_DIR / "English-Bangla.txt"
SYL_BN_CSV = RAW_DATA_DIR / "Sylheti-Bangla.csv"

OUTPUT_CSV = PROCESSED_DATA_DIR / "train.csv"


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
# DATA LOADERS
# ==================================================

def load_en_bn_txt(path: Path) -> pd.DataFrame:
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 3:
                continue

            en = normalize_english(parts[0])
            bn = normalize_bangla(parts[1])

            data.extend(
                [
                    {
                        "input": f"Translate English to Bangla: {en}",
                        "output": bn,
                    },
                    {
                        "input": f"Translate Bangla to English: {bn}",
                        "output": en,
                    },
                ]
            )

    return pd.DataFrame(data)


def load_syl_bn_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    data = []

    for _, row in df.iterrows():
        syl = normalize_sylheti(row["sylheti"])
        bn = normalize_bangla(row["bangla"])

        data.extend(
            [
                {
                    "input": f"Translate Sylheti to Bangla: {syl}",
                    "output": bn,
                },
                {
                    "input": f"Translate Bangla to Sylheti: {bn}",
                    "output": syl,
                },
            ]
        )

    return pd.DataFrame(data)


# ==================================================
# DATASET BUILDING
# ==================================================

def build_dataset() -> pd.DataFrame:
    df_syl = load_syl_bn_csv(SYL_BN_CSV)
    df_en = load_en_bn_txt(EN_BN_TXT)

    target_size = len(df_syl)

    df_en = df_en.sample(
        n=target_size,
        random_state=42,
    )

    final_df = pd.concat(
        [df_syl, df_en],
        ignore_index=True,
    )

    final_df = final_df.drop_duplicates()

    final_df = (
        final_df.sample(
            frac=1,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    print(f"EN-BN size : {len(df_en):,}")
    print(f"SYL-BN size: {len(df_syl):,}")
    print(f"Final size : {len(final_df):,}")

    return final_df


# ==================================================
# MAIN
# ==================================================

def main() -> None:
    final_df = build_dataset()

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    print(f"\nSaved dataset to:")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()