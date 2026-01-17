import click
import pandas as pd
from utils.arabic_text import clean_arabic_text
from utils.arabic_text import clean_arabic_text, remove_stopwords, normalize_arabic

@click.group(help="Arabic text preprocessing commands")
def preprocess_group():
    pass


@preprocess_group.command("remove")
@click.option("--csv_path", required=True, type=click.Path(exists=True))
@click.option("--text_col", required=True, type=str)
@click.option("--output", required=True, type=str)
def remove(csv_path, text_col, output):
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise click.ClickException(f"Column '{text_col}' not found")

    df[text_col] = df[text_col].astype(str).apply(clean_arabic_text)
    df.to_csv(output, index=False)

    click.echo(f" Cleaned file saved to {output}")
from utils.arabic_text import remove_stopwords  # ضيفيها مع الاستيرادات فوق

@preprocess_group.command("stopwords")
@click.option("--csv_path", required=True, type=click.Path(exists=True))
@click.option("--text_col", required=True, type=str)
@click.option("--output", required=True, type=str)
def stopwords(csv_path, text_col, output):
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise click.ClickException(f"Column '{text_col}' not found")

    df[text_col] = df[text_col].astype(str).apply(remove_stopwords)
    df.to_csv(output, index=False)

    click.echo(f"Stopwords removed. Saved to {output}")
@preprocess_group.command("replace")
@click.option("--csv_path", required=True, type=click.Path(exists=True))
@click.option("--text_col", required=True, type=str)
@click.option("--output", required=True, type=str)
def replace(csv_path, text_col, output):
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise click.ClickException(f"Column '{text_col}' not found")

    df[text_col] = df[text_col].astype(str).apply(normalize_arabic)
    df.to_csv(output, index=False)

    click.echo(f" Arabic text normalized. Saved to {output}")
@preprocess_group.command("all")
@click.option("--csv_path", required=True, type=click.Path(exists=True))
@click.option("--text_col", required=True, type=str)
@click.option("--output", required=True, type=str)
def all_steps(csv_path, text_col, output):
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise click.ClickException(f"Column '{text_col}' not found")

    # 1) remove
    df[text_col] = df[text_col].astype(str).apply(clean_arabic_text)
    # 2) stopwords
    df[text_col] = df[text_col].astype(str).apply(remove_stopwords)
    # 3) replace/normalize
    df[text_col] = df[text_col].astype(str).apply(normalize_arabic)

    df.to_csv(output, index=False)
    click.echo(f"✅ Preprocessing ALL done. Saved to {output}")

