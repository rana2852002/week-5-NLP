from __future__ import annotations
from pathlib import Path
import click
import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from model2vec import StaticModel

# TF-IDF
def build_tfidf(
    texts: list[str],
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b"
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def save_tfidf(X, vectorizer, out_dir: str | Path, prefix: str = "tfidf"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = out_dir / f"{prefix}_vectors.npz"
    vec_path = out_dir / f"{prefix}_vectorizer.pkl"

    sp.save_npz(X_path, X)
    joblib.dump(vectorizer, vec_path)
    return X_path, vec_path


# ----------------------------
# Model2Vec (ARBERTv2)
# ----------------------------
def build_model2vec(
    texts: list[str],
    model_name: str = "JadwalAlmaa/model2vec-ARBERTv2",
):
    model = StaticModel.from_pretrained(model_name)
    emb = model.encode(texts)
    return emb, model


def save_model2vec(
    embeddings: np.ndarray,
    model: StaticModel,
    out_dir: str | Path,
    prefix: str = "model2vec"
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / f"{prefix}_embeddings.npy"
    model_path = out_dir / f"{prefix}_model.pkl"

    np.save(emb_path, embeddings)
    joblib.dump(model, model_path)
    return emb_path, model_path


@click.group(help="Embedding commands")
def embed_group():
    pass


@embed_group.command("tfidf")
@click.option("--csv_path", required=True, type=click.Path(exists=True))
@click.option("--text_col", required=True)
@click.option("--output_dir", default="outputs/embeddings")
def tfidf_cmd(csv_path, text_col, output_dir):
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()

    X, vectorizer = build_tfidf(texts)

    # يحفظ النسخ الأساسية (NPZ + Vectorizer)
    save_tfidf(X, vectorizer, output_dir)

    # ✅ يحفظ نسخة متوافقة مع training (PKL: {"X": X, "vectorizer": vectorizer})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"X": X, "vectorizer": vectorizer}, out_dir / "tfidf_vectors.pkl")

    click.echo("✅ TF-IDF embeddings saved")


@embed_group.command("model2vec")
@click.option("--csv_path", required=True, type=click.Path(exists=True))
@click.option("--text_col", required=True)
@click.option("--output_dir", default="outputs/embeddings")
def model2vec_cmd(csv_path, text_col, output_dir):
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()

    embeddings, model = build_model2vec(texts)
    save_model2vec(embeddings, model, output_dir)

    click.echo("✅ Model2Vec embeddings saved")
