# Sylheti Translator: Sylheti → Bangla → English

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

## Overview

This project is a **custom translation pipeline** for the low-resource Sylheti language.  
It translates Sylheti text to English by first converting **Sylheti → Bangla** using a fine-tuned **uMT5-small model**, then **Bangla → English** using [Argos Translate](https://www.argosopentech.com/).

This demonstrates **low-resource NLP modeling**, preprocessing, and building an end-to-end translation pipeline — all fully reproducible.

---

## Features

- **Sylheti → Bangla translation** using a fine-tuned uMT5-small model  
- **Bangla → English translation** via Argos Translate  
- Fully functional **Python pipeline**
- Easy to reproduce and extend with additional datasets

---

## Dataset

This project uses the **Sylheti–Bangla parallel dataset** provided by Tabia Tanzin Prama and Mangsura Kabir Oni in [A Dataset for Translating Local Bangla (Sylheti) Dialects into Standard Bangla](https://www.sciencedirect.com/science/article/pii/S2352340926001290).

- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
- **Usage:** Free to share, copy, and modify for research or commercial purposes with attribution.  
- **Modifications:** Preprocessed to create parallel text pairs suitable for training a Sylheti → Bangla model.

**Example entry (after preprocessing):**

| Sylheti        | Bangla           |
|----------------|-----------------|
| ami zaimu      | আমি যাব          |
| tumi koi zaiba | তুমি কোথায় যাবে |

> The full dataset used is included in this repository.

---

## Model

- **Architecture:** [uMT5-small](https://huggingface.co/google/umt5-small)  
- **Preprocessing:** Tokenized with SentencePiece, padding masked with `-100`  
- **Training:** Fine-tuned on ~5000 Sylheti–Bangla pairs (expanded to ~10,000 with bidirectional pairs)  
- **Learning Rate:** 5e-5  
- **Device:** GPU-enabled (tested on NVIDIA RTX 4070)

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
- argostranslate  
- torch==2.5.1+cu121  

---

Run setup scripts:

```bash
# Setup Argos Translate
python argos_setup.py

# Preprocess the dataset
python preprocessing.py
```

---

## Training

Make sure GPU is active (optional if using CPU):

```bash
python test_gpu.py
# Prints True if GPU is active, also prints torch version

# Train the model and save it to umt5_sylheti_model/
python train.py
```

Quick test training is available via **quick_train.py** using only the first 500 pairs.

---

## Inference
```bash
python inference.py --text "ami haitam sai"
# Runs Sylheti → Bangla translation and full Sylheti → English pipeline
# Defaults to "ami zaimu" if --text is not provided
```

---

## Folder Structure

Sylheti-Translator/  
│  
├── data/  
│   └── Syl-Ban.csv  
├── .gitignore  
├── requirements.txt  
├── argos_setup.py  
├── detect_script.py  
├── inference.py  
├── preprocessing.py  
├── quick_train.py  
├── README.md  
├── Sylheti-Translator.ipynb  
├── test_gpu.py  
├── train.py  
└── transliterate_latin_sylheti.py  

Automatically generated folders from training -- umt5_sylheti_model (contains output) and umt5_sylheti_model_sanity are excluded from Git via **.gitignore.**

---

## Future Work

- Train direct Sylheti-English model to remove intermediate Bangla step  
- Expand dataset with additional Sylheti text  
- Add BLEU / ROUGE evaluation metrics  
- Deploy as a web app for real-time translation  
- Implement picture (image) and audio processing for multimodal Sylheti input  

---

## License

Code: Nihal Uddin  
Dataset: CC BY 4.0  
 by Tabia Tanzin Prama & Mangsura Kabir Oni  

---