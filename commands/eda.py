import os
import click
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join("outputs", "visualizations")


def _ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


@click.group(help="Exploratory Data Analysis (EDA) commands")
def eda_group():
    pass


@eda_group.command("distribution", help="Plot class distribution (pie or bar)")
@click.option("--csv_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--label_col", required=True, type=str)
@click.option("--plot_type", default="pie", type=click.Choice(["pie", "bar"], case_sensitive=False))
def distribution(csv_path, label_col, plot_type):
    _ensure_dirs()
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise click.ClickException(f"Column '{label_col}' not found. Available: {list(df.columns)}")

    counts = df[label_col].value_counts(dropna=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if plot_type.lower() == "pie":
        ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
        ax.set_title(f"Class Distribution (pie) - {label_col}")
    else:
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"Class Distribution (bar) - {label_col}")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    out_path = os.path.join(OUTPUT_DIR, f"class_distribution_{label_col}_{plot_type.lower()}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    click.echo("Class distribution saved:")
    click.echo(f"   {out_path}")
    click.echo(f" Summary: total={len(df)}, classes={counts.shape[0]}")


@eda_group.command("histogram", help="Plot text length histogram (words or chars)")
@click.option("--csv_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--text_col", required=True, type=str)
@click.option("--unit", default="words", type=click.Choice(["words", "chars"], case_sensitive=False))
def histogram(csv_path, text_col, unit):
    _ensure_dirs()
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise click.ClickException(f"Column '{text_col}' not found. Available: {list(df.columns)}")

    texts = df[text_col].fillna("").astype(str)

    if unit.lower() == "words":
        lengths = texts.apply(lambda s: len(s.split()))
        title = "Text Length (words)"
    else:
        lengths = texts.apply(len)
        title = "Text Length (chars)"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(lengths.values, bins=30)
    ax.set_title(title)
    ax.set_xlabel("Length")
    ax.set_ylabel("Frequency")

    out_path = os.path.join(OUTPUT_DIR, f"text_length_{text_col}_{unit.lower()}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    click.echo("Histogram saved:")
    click.echo(f"   {out_path}")
    click.echo(f" Stats: mean={lengths.mean():.2f}, median={lengths.median():.2f}, min={lengths.min()}, max={lengths.max()}")

