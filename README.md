# Sylheti Translator: Sylheti → Bangla → English

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

## Overview

This project is a **custom translation pipeline** for the low-resource Sylheti language.  
It translates Sylheti text to English by first converting **Sylheti → Bangla** using a fine-tuned **uMT5-small model**, then **Bangla → English** ...

This demonstrates **low-resource NLP modeling**, preprocessing, and building an end-to-end translation pipeline — all fully reproducible.

---

## Features

- **Sylheti → Bangla translation** and **Bangla → English translation** using a fine-tuned uMT5-small model
- Fully functional **Python pipeline**
- Easy to reproduce and extend with additional datasets

---

## Dataset

This project uses the **Sylheti–Bangla parallel dataset** provided by Tabia Tanzin Prama and Mangsura Kabir Oni in [A Dataset for Translating Local Bangla (Sylheti) Dialects into Standard Bangla](https://www.sciencedirect.com/science/article/pii/S2352340926001290), as well as the **English-Bangla parallel dataset** provided by [www.manythings.org/anki](https://www.manythings.org/anki/) and [tatoeba.org](http://tatoeba.org/home).

- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
- **Usage:** Free to share, copy, and modify for research or commercial purposes with attribution.

**Example entry (after preprocessing):**

| Sylheti        | Bangla           | English              |
|----------------|-----------------|---------------------|
| ami zaimu      | আমি যাব          | i will go           |
| tumi koi zaiba | তুমি কোথায় যাবে | where will you go?  |

> The full datasets used are included in this repository.

---

## Model

- **Architecture:** [uMT5-small](https://huggingface.co/google/umt5-small)  
- **Preprocessing:** Tokenized with SentencePiece, padding masked with `-100`  
- **Training:** Fine-tuned on ~5000 Sylheti–Bangla pairs (expanded to ~10,000 with bidirectional pairs)  
- **Learning Rate:** 5e-5  
- **Device:** GPU-enabled (tested on NVIDIA GeForce RTX 4070 Laptop GPU)

---

## Installation and Setup

Create a virtual environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

requirements.txt contents:  
- pandas  
- transformers==4.40.2  
- datasets  
- sentencepiece  
- accelerate  
- torch==2.5.1+cu121  

---

Run setup scripts:

```bash
# Preprocess the dataset
python -m translator.preprocessing
```

---

## Training

```bash
# Make sure GPU is active (optional if using CPU) -- prints True if it is active, as well as torch version:
python -m training.test_gpu

# Train the Sylheti-Bangla model and save it to umt5_sylheti_model/
python -m training.train
```

Test training is available via **quick_train.py** in the training folder, which using 500 random pairs.

---

## Inference
```bash
python inference.py --text "ami haitam sai"
# Runs Sylheti → Bangla translation and full Sylheti → English pipeline
# Defaults to "আমি যাইমু" ("ami zaimu") if text parameter is not provided in the terminal
```

---

## Folder Structure

Sylheti-Translator/  
├── data/  
│   ├── raw/  
│   │   ├── English-Bangla.txt  
│   │   └── Sylheti-Bangla.csv  
│   └── processed/  
├── scripts/  
│   ├── build_dataset.py  
├── training/  
│   ├── quick_train.py   
│   ├── test_gpu.py  
│   └── train.py  
├── translator/  
│   ├── detect_script.py  
│   ├── inference.py  
│   └── transliterate_latin_sylheti.py  
├── .gitignore  
├── main.py  
├── README.md  
└── requirements.txt  

Automatically generated folders from training -- umt5_sylheti_model (contains output) and umt5_sylheti_model_sanity are excluded from Git via **.gitignore.**

---

## Future Work

- Accept Sylheti input using Latin alphabet
- Train direct Sylheti-English model to remove intermediate Bangla step  
- Expand dataset with additional Sylheti text  
- Add BLEU / ROUGE evaluation metrics  
- Deploy as a web app for real-time translation  
- Implement picture (image) and audio processing for multimodal Sylheti input  

---

## License

Code: Nihal Uddin  
Dataset: CC BY 4.0  
 Sylheti-Bangla by Tabia Tanzin Prama & Mangsura Kabir Oni
 English-Bangla by [www.manythings.org/anki](https://www.manythings.org/anki/) & [tatoeba.org](http://tatoeba.org/home)

---