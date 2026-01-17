import os
import json
import click
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def ensure_outputs():
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)


def smart_split(X, y, base_test_size=0.2, random_state=42):
    n_samples = len(y)
    n_classes = y.nunique()

    test_size = base_test_size
    min_test = n_classes
    calc_test = int(round(n_samples * test_size))

    if calc_test < min_test:
        test_size = min(0.5, max(test_size, min_test / n_samples))

    use_stratify = True
    if y.value_counts().min() < 2:
        use_stratify = False

    kwargs = dict(test_size=test_size, random_state=random_state)
    if use_stratify:
        kwargs["stratify"] = y

    return train_test_split(X, y, **kwargs)


def get_models_for_space(space: str):
    """
    space: 'tfidf' or 'dense'
    - TF-IDF (sparse): LogisticRegression, MultinomialNB, KNN
    - Dense (Model2Vec): LogisticRegression, KNN, RandomForest
    """
    if space == "tfidf":
        return {
            "lr": LogisticRegression(max_iter=3000),
            "nb": MultinomialNB(),
            "knn": KNeighborsClassifier(n_neighbors=5),
        }
    else:
        return {
            "lr": LogisticRegression(max_iter=3000),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        }


def md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def build_markdown_report(title: str, results: list[dict], out_path: str, notes: str = ""):
    lines = []
    lines.append(f"# {title}\n")
    if notes:
        lines.append(f"> {notes}\n")

    lines.append("## Summary\n")
    lines.append("| Model | Accuracy | Saved Model | Report |\n")
    lines.append("|---|---:|---|---|\n")
    for r in results:
        lines.append(
            f"| {md_escape(r['model_name'])} | {r['accuracy']:.4f} | `{r['model_path']}` | `{r['txt_report_path']}` |\n"
        )

    # Best model
    best = max(results, key=lambda x: x["accuracy"])
    lines.append("\n## Best Model\n")
    lines.append(f"- **{best['model_name']}** with accuracy **{best['accuracy']:.4f}**\n")

    # Add per-model details (short)
    lines.append("\n## Details (per model)\n")
    for r in results:
        lines.append(f"\n### {r['model_name']}\n")
        lines.append(f"- Accuracy: **{r['accuracy']:.4f}**\n")
        lines.append(f"- Model file: `{r['model_path']}`\n")
        lines.append(f"- Full classification report: `{r['txt_report_path']}`\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


@click.group(help="Train models and generate reports")
def train_group():
    pass


def run_training(
    X,
    y: pd.Series,
    models_to_run: list[str],
    space: str,  # "tfidf" or "dense"
    report_prefix: str,
):
    ensure_outputs()

    X_train, X_test, y_train, y_test = smart_split(X, y)

    available = get_models_for_space(space)

    # Validate requested models
    bad = [m for m in models_to_run if m not in available]
    if bad:
        raise click.ClickException(
            f"Unknown model(s): {bad}. Allowed for {space}: {list(available.keys())}"
        )

    results = []

    for key in models_to_run:
        model = available[key]
        model_name = {
            "lr": "LogisticRegression",
            "nb": "MultinomialNB",
            "knn": "KNN",
            "rf": "RandomForest",
        }.get(key, key)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        txt_report = classification_report(y_test, y_pred, zero_division=0)

        model_path = f"outputs/models/{report_prefix}_{key}.pkl"
        txt_report_path = f"outputs/reports/{report_prefix}_{key}.txt"

        joblib.dump(model, model_path)
        with open(txt_report_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL: {model_name}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(txt_report)

        results.append(
            {
                "key": key,
                "model_name": model_name,
                "accuracy": acc,
                "model_path": model_path,
                "txt_report_path": txt_report_path,
            }
        )

    # Write markdown summary for this run
    md_path = f"outputs/reports/{report_prefix}_summary.md"
    notes = "Auto split adapts for small datasets; metrics may be unstable with tiny data."
    build_markdown_report(
        title=f"Training Report ({report_prefix})",
        results=results,
        out_path=md_path,
        notes=notes,
    )

    return results, md_path


@train_group.command("tfidf")
@click.option("--vectors_path", required=True, type=click.Path(exists=True))
@click.option("--csv_path", required=True, type=click.Path(exists=True))
@click.option("--label_col", required=True, type=str)
@click.option(
    "--models",
    default="lr,nb,knn",
    help="Comma-separated: lr,nb,knn",
)
def train_tfidf(vectors_path, csv_path, label_col, models):
    # Load vectors
    if vectors_path.endswith(".pkl"):
        payload = joblib.load(vectors_path)
        X = payload["X"]
    elif vectors_path.endswith(".npz"):
        X = sp.load_npz(vectors_path)
    else:
        raise click.ClickException("vectors_path must be .pkl or .npz")

    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise click.ClickException(f"Column '{label_col}' not found")
    y = df[label_col].astype(str)

    models_to_run = [m.strip() for m in models.split(",") if m.strip()]
    results, md_path = run_training(
        X=X,
        y=y,
        models_to_run=models_to_run,
        space="tfidf",
        report_prefix="tfidf",
    )

    best = max(results, key=lambda x: x["accuracy"])
    click.echo(" TF-IDF training done")
    click.echo(f"Best: {best['model_name']} acc={best['accuracy']:.4f}")
    click.echo(f"Markdown summary: {md_path}")


@train_group.command("model2vec")
@click.option("--emb_path", required=True, type=click.Path(exists=True))
@click.option("--csv_path", required=True, type=click.Path(exists=True))
@click.option("--label_col", required=True, type=str)
@click.option(
    "--models",
    default="lr,knn,rf",
    help="Comma-separated: lr,knn,rf",
)
def train_model2vec(emb_path, csv_path, label_col, models):
    X = np.load(emb_path)

    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise click.ClickException(f"Column '{label_col}' not found")
    y = df[label_col].astype(str)

    models_to_run = [m.strip() for m in models.split(",") if m.strip()]
    results, md_path = run_training(
        X=X,
        y=y,
        models_to_run=models_to_run,
        space="dense",
        report_prefix="model2vec",
    )

    best = max(results, key=lambda x: x["accuracy"])
    click.echo(" Model2Vec training done")
    click.echo(f"Best: {best['model_name']} acc={best['accuracy']:.4f}")
    click.echo(f"Markdown summary: {md_path}")
