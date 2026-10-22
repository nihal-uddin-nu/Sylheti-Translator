# Sylheti Translator: Low-Resource NLP and Machine Translation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

A Sylheti-focused language-processing toolkit combining neural machine translation, transliteration, script detection, and language-input tools. The project supports translation workflows involving **Sylheti, standard Bangla, and English**, with an emphasis on improving access to and support for the low-resource Sylheti language.

## Overview

Sylheti is a low-resource language variety with limited parallel translation data and variation in how it is represented in writing. This project explores how multilingual neural machine translation, parallel corpora, and language-processing algorithms can be combined to address these challenges.

The system supports translation workflows among Sylheti, Bangla, and English. Its primary focus is translating Sylheti to and from English, using Bangla as an intermediate language where appropriate. Bangla is also supported as an input and output language in its own right.

Beyond translation, the project includes utilities for script detection and transliteration between Latin and Bengali scripts. Trie-based autocomplete is under development, while spelling correction is planned but has not yet entered development.

The goal is to build a modular Sylheti NLP toolkit that combines practical translation capabilities with future improvements to text input and language processing.

## Features

- **Sylheti ↔ English translation:** Translate between Sylheti and English through an intermediate Bangla representation.
- **Sylheti ↔ Bangla translation:** Support translation between Sylheti and standard Bangla.
- **Bangla ↔ English translation:** Support Bangla as both a source and target language.
- **Script detection:** Identify the script used in input text.
- **Sylheti transliteration:** Provide utilities and mapping resources for converting between Latin-alphabet Sylheti and Bengali-script Sylheti.
- **Autocomplete — work in progress:** Develop word- and sentence-level suggestions using trie-based data structures for Sylheti, Bangla, and English.
- **Spelling correction — not started:** Planned functionality for detecting and correcting misspellings in Sylheti text.
- **Neural machine translation:** Fine-tune uMT5-small for Sylheti-to-Bangla translation.
- **GPU acceleration:** Support training with compatible CUDA-enabled hardware.

## Translation Architecture

Bangla serves as a bridge between Sylheti and English, allowing the project to leverage existing Bangla–English parallel data.

### Sylheti → English

1. Accept Sylheti input.
2. Translate Sylheti into standard Bangla.
3. Translate the Bangla result into English.

### English → Sylheti

1. Accept English input.
2. Translate English into Bangla.
3. Translate the Bangla result into Sylheti.

### Sylheti ↔ Bangla

Translate between Sylheti and standard Bangla using the corresponding translation functionality.

### Bangla ↔ English

Translate between Bangla and English directly, without requiring the complete Sylheti translation pipeline.

The exact model and inference path used for each direction depend on the implementation in the translation modules.

## Dataset

The project uses two parallel datasets.

### Sylheti–Bangla Parallel Corpus

The Sylheti–Bangla dataset is provided by Tabia Tanzin Prama and Mangsura Kabir Oni in:

[A Dataset for Translating Local Bangla (Sylheti) Dialects into Standard Bangla](https://www.sciencedirect.com/science/article/pii/S2352340926001290)

This corpus provides the principal training data for the Sylheti-to-Bangla translation model.

### English–Bangla Parallel Corpus

English–Bangla sentence pairs are sourced from:

- [ManyThings.org — English–Bangla data](https://www.manythings.org/anki/)
- [Tatoeba](https://tatoeba.org/)

These datasets provide parallel examples for the Bangla–English translation stage.

### Example Translations

| Sylheti | Standard Bangla | English |
|---|---|---|
| ami zaimu | আমি যাব | I will go |
| tumi koi zaiba | তুমি কোথায় যাবে | Where will you go? |

*Examples are illustrative and are not guaranteed model outputs.*

### Data Preparation

Raw datasets are stored in `data/raw/`, while processed datasets are stored in `data/processed/`.

The preparation scripts handle dataset processing and the generation of supporting resources.

Run the setup workflow from the repository root:

```bash
python -m scripts.setup
```

This invokes the setup entry point, which coordinates the configured dataset-building and trie-generation workflow.

### Dataset Licensing and Attribution

The original Sylheti–Bangla publication and the English–Bangla data providers must be credited according to their applicable terms. The license of one dataset does not automatically apply to all other datasets in the repository.

Consult the original sources before redistributing data or using it commercially.

## Model and Training

### Sylheti–Bangla Translation Model

The primary neural translation model uses [Google uMT5-small](https://huggingface.co/google/umt5-small), a multilingual sequence-to-sequence architecture.

| Setting | Value |
|---|---|
| Architecture | uMT5-small |
| Framework | PyTorch |
| Tokenization | SentencePiece |
| Learning rate | `5e-5` in the documented configuration |
| Training data | Approximately 5,000 Sylheti–Bangla pairs |
| Training hardware | GPU acceleration supported |

The training data may be expanded with bidirectional examples where appropriate. The final dataset size and model configuration depend on the preprocessing and training scripts.

The Bangla–English translation stage uses the corresponding translation functionality implemented in the project. Refer to the source code for its current model configuration.

## Installation and Setup

Python 3.11 is recommended.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Sylheti-Translator
```

Replace `<repository-url>` with the actual repository URL.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project dependencies include packages such as:

- `pandas`
- `transformers`
- `datasets`
- `sentencepiece`
- `accelerate`
- `torch`

See `requirements.txt` for the exact pinned versions.

**PyTorch note:** The appropriate PyTorch build depends on your operating system, GPU, and CUDA compatibility. A CUDA-specific package may need to be replaced with a compatible build if you are training on CPU or using a different CUDA configuration.

### 4. Build Datasets and Supporting Resources

```bash
python -m scripts.setup
```

Run this command after cloning the repository to generate the processed datasets and supporting resources required by the configured workflow.

Autocomplete resources are generated locally where the required source data and scripts are available. Autocomplete itself remains a work in progress.

## Training

### Check GPU Availability

```bash
python -m training.utils.test_gpu
```

This diagnostic checks whether PyTorch can access a CUDA-enabled GPU.

### Full Training Run

```bash
python -m training.translation.train
```

This launches the main Sylheti–Bangla model training script.

### Quick Training Run

```bash
python -m training.translation.quick_train
```

The quick-training script is intended for testing the training workflow with a smaller subset of the data. It is not a substitute for a full training run.

Consult the training scripts for the current training parameters, checkpoint locations, and dataset requirements.

## Inference

The inference module is located at `translator/inference.py`, and `main.py` serves as the main application entry point.

Launch the application with:

```bash
python main.py
```

For direct command-line inference, an example invocation is:

```bash
python -m translator.inference --text "ami haitam sai"
```

This command assumes that `translator/inference.py` implements an `--text` argument. If the module exposes only Python functions, use its implemented interface or add a command-line argument parser.

The available input and output languages depend on the inference paths supported by the current implementation.

## Transliteration and Script Detection

Sylheti can be entered using different writing conventions. The project separates script detection and transliteration from the neural translation model so that these capabilities can be developed independently.

### Latin-Alphabet Sylheti → Bengali Script

The module:

```text
translator/transliterate_latin_sylheti.py
```

provides the Latin-to-Bengali transliteration utility.

Its mapping resources are divided into:

- `vowels.json` — vowel mappings
- `consonants.json` — consonant mappings
- `conjuncts.json` — conjunct mappings

This component helps convert Latin-alphabet input into the script representation expected by the translation workflow.

### Bengali Script → Latin Alphabet

The reverse-direction mapping resource is stored at:

```text
resources/transliteration/bengali_script_to_latin/mappings.json
```

These mappings support transliteration from Bengali script into Latin characters.

### Script Detection

The module:

```text
translator/detect_script.py
```

handles script detection. It provides a dedicated location for identifying the writing system used in input text and routing text through appropriate processing steps.

The precise behavior depends on the implementation and available mapping resources.

## Autocomplete and Spelling Correction

### Autocomplete — Work in Progress

Autocomplete is an active work-in-progress component of the project. The current infrastructure includes scripts for building trie-based resources intended to support word- and sentence-level suggestions in Sylheti, Bangla, and English.

The relevant files are:

```text
resources/autocomplete/
scripts/build_tries.py
```

The autocomplete resources are generated from the available source data using the repository's setup workflow:

```bash
python -m scripts.setup
```

The generated trie contents are excluded from Git through `.gitignore` to avoid committing generated word- and sentence-level resources.

**Autocomplete is not yet a finished, fully integrated feature.** The existing infrastructure provides a foundation for development rather than a guarantee of complete, user-facing autocomplete functionality.

Potential improvements include:

- Prefix-based word suggestions.
- Sentence completion.
- Context-aware next-word prediction.
- Part-of-speech-aware suggestions.
- Better handling of spelling variation in Latin-alphabet Sylheti.

### Spelling Correction — Not Started

Spelling correction has **not been implemented, and development has not started**. The `resources/spellcheck/` directory is reserved for future spelling-related resources.

Potential future work includes:

- Detecting common Sylheti misspellings.
- Phonetic matching to suggest likely corrections.
- Accounting for spelling variation in Latin-alphabet Sylheti.
- Using surrounding context to rank candidate corrections.

These are planned directions, not existing capabilities.

## Project Structure

```text
Sylheti-Translator/
├── data/
│   ├── processed/
│   └── raw/
│       ├── English-Bangla.txt
│       └── Sylheti-Bangla.csv
│
├── resources/
│   ├── autocomplete/
│   │   └── Generated word and sentence tries
│   ├── spellcheck/
│   └── transliteration/
│       ├── bengali_script_to_latin/
│       │   └── mappings.json
│       └── latin_to_bengali_script/
│           ├── conjuncts.json
│           ├── consonants.json
│           └── vowels.json
│
├── scripts/
│   ├── build_datasets.py
│   ├── build_tries.py
│   └── setup.py
│
├── training/
│   ├── translation/
│   │   ├── quick_train.py
│   │   └── train.py
│   ├── transliteration/
│   └── utils/
│       └── test_gpu.py
│
├── translator/
│   ├── detect_script.py
│   ├── inference.py
│   └── transliterate_latin_sylheti.py
│
├── main.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Directory Responsibilities

| Path | Purpose |
|---|---|
| `data/raw/` | Original parallel datasets |
| `data/processed/` | Processed training and application data |
| `resources/autocomplete/` | Generated autocomplete tries; functionality is a work in progress |
| `resources/spellcheck/` | Reserved for future spelling-correction resources; development has not started |
| `resources/transliteration/` | Transliteration mappings |
| `scripts/` | Dataset preparation, setup orchestration, and trie generation |
| `training/translation/` | Translation training scripts |
| `training/transliteration/` | Transliteration-related training code |
| `training/utils/` | Training utilities and diagnostics |
| `translator/` | Inference, script detection, and transliteration |
| `main.py` | Main application entry point |

### Generated Files and Git

Generated autocomplete resources are excluded from Git using `.gitignore`. Their contents are omitted from the repository to avoid committing generated word- and sentence-level tries for Sylheti, Bangla, and English.

Model-training output directories, including `umt5_sylheti_model/` and `umt5_sylheti_model_sanity/`, are also excluded.

Consequently, cloning the repository may not provide generated resources or trained model checkpoints. Run the setup workflow and obtain or recreate the necessary model artifacts before using functionality that depends on them.

## Technology Stack

- **Python 3.11** — application logic and data processing
- **PyTorch** — model training and inference
- **Hugging Face Transformers** — model loading and fine-tuning
- **uMT5-small** — multilingual sequence-to-sequence translation
- **SentencePiece** — subword tokenization
- **Pandas** — dataset processing
- **JSON** — transliteration mappings
- **Trie data structures** — autocomplete infrastructure

## Future Work

Future development will focus on improving Sylheti language processing and completing features that are still in progress.

### Translation

- Train and evaluate a direct Sylheti–English model to compare against the existing two-stage pipeline.
- Expand the Sylheti–Bangla parallel corpus.
- Evaluate translation quality using BLEU, chrF, and human evaluation.
- Investigate how the intermediate Bangla translation affects meaning preservation.

### Autocomplete — In Progress

- Complete and integrate trie-based autocomplete.
- Improve word- and sentence-level suggestions.
- Incorporate part-of-speech information and contextual language modeling.
- Evaluate suggestion quality and response time.

### Spelling Correction — Not Started

- Design and implement a spelling-correction pipeline.
- Explore phonetic matching for Latin-alphabet Sylheti.
- Develop candidate ranking using contextual information.
- Evaluate correction accuracy on a manually reviewed test set.

### Transliteration and Application Development

- Improve transliteration coverage for spelling and pronunciation variation.
- Develop a web interface for translation, transliteration, and autocomplete.
- Explore speech recognition and image-based text input for Sylheti.

## License and Attribution

### Code

Copyright © Nihal Uddin.

The code license should be specified in a separate `LICENSE` file. The repository's code license and the licenses of third-party datasets are distinct.

### Sylheti–Bangla Dataset

Tabia Tanzin Prama and Mangsura Kabir Oni, *A Dataset for Translating Local Bangla (Sylheti) Dialects into Standard Bangla*.

[Original publication](https://www.sciencedirect.com/science/article/pii/S2352340926001290)

### English–Bangla Data

- [ManyThings.org](https://www.manythings.org/anki/)
- [Tatoeba](https://tatoeba.org/)

Please consult the original sources for their applicable licensing, attribution, and redistribution requirements.

---

**A Sylheti-focused NLP project exploring machine translation, transliteration, and language-input tools for a low-resource language.**