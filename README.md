# Arabic NLP CLI Pipeline

A modular, reproducible Command-Line Interface (CLI) for Arabic text classification.

This project implements a **full NLP pipeline** that can be applied to **any unseen dataset**
without changing the source code.  
The focus is on **correct ML workflow design**, not just model accuracy.

---

## Problem

- **Task:** Arabic text classification (binary or multi-class)
- **Input:** Raw Arabic text
- **Output:** Predicted class labels
- **Goal:** Build a clean, reusable NLP pipeline that works on instructor-provided datasets

---

## Project Philosophy

In Natural Language Processing, strong results alone are not sufficient.  
What truly matters is building a pipeline that is:

- **Dataset-agnostic**
- **Modular**
- **Reproducible**
- **Easy to evaluate and extend**

The pipeline strictly separates:
- Exploratory Data Analysis (EDA)
- Text preprocessing
- Feature extraction (embeddings)
- Model training and evaluation

All steps are executed via CLI commands.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Dataset Requirements

The dataset must be provided as a **CSV file** containing:

- **One text column** (e.g. `review`, `text`, `description`)
- **One label column** (e.g. `sentiment`, `class`)

If the dataset is not in CSV format, it should be converted before running the pipeline.

**Note:** The CLI expects CSV input.


Before running the commands, replace:

- `<DATASET>` → Path to the CSV file  
- `<TEXT_COL>` → Name of the text column  
- `<LABEL_COL>` → Name of the label column  


## 1. Exploratory Data Analysis (EDA)
```bash
python main.py eda distribution \
  --csv_path <DATASET> \
  --label_col <LABEL_COL>

python main.py eda histogram \
  --csv_path <DATASET> \
  --text_col <TEXT_COL>

```

**Outputs:**
- Class distribution plot
- Text length histogram  
Saved in: `outputs/visualizations/`


## 2. Text Preprocessing
```bash
python main.py preprocess all \
  --csv_path <DATASET> \
  --text_col <TEXT_COL> \
  --output data/final_cleaned.csv

```
## Includes:
- Text cleaning
- Arabic normalization
- Stopword removal

## 3. Embeddings

### TF-IDF

```bash
python main.py embed tfidf \
  --csv_path data/final_cleaned.csv \
  --text_col <TEXT_COL>
  ```
### Model2Vec (ARBERTv2)
```bash
python main.py embed model2vec \
  --csv_path data/final_cleaned.csv \
  --text_col <TEXT_COL>
 ```
 ## Embeddings are saved in: outputs/embeddings/

 ## 4. Training & Evaluation

## TF-IDF Training
This step supports training multiple classical classifiers.


```bash
python main.py train tfidf \
  --vectors_path outputs/embeddings/tfidf_vectors.pkl \
  --csv_path data/final_cleaned.csv \
  --label_col <LABEL_COL> \
  --models lr,nb,knn
  ```
## Model2Vec
```bash
python main.py train model2vec \
  --emb_path outputs/embeddings/model2vec_embeddings.npy \
  --csv_path data/final_cleaned.csv \
  --label_col <LABEL_COL>
  ```
## Outputs
- Models: `outputs/models/`
- Reports (Markdown): `outputs/reports/`
- Visualizations: `outputs/visualizations/`


## Notes on Small Datasets

The training pipeline includes a **smart data splitting strategy**.

If the dataset is very small:
- The test split size is automatically adjusted
- Stratified splitting is disabled when class counts are insufficient

This ensures that the pipeline runs **without errors**, even on small instructor-provided datasets.