import click
from commands.eda import eda_group
from commands.preprocessing import preprocess_group
from commands.embedding import embed_group
from commands.training import train_group



@click.group(help="Arabic NLP Classification CLI Tool: EDA → Preprocess → Embedding → Training")
def cli():
    pass


cli.add_command(eda_group, name="eda")
cli.add_command(preprocess_group, name="preprocess")
cli.add_command(embed_group, name="embed")
cli.add_command(train_group, name="train")


if __name__ == "__main__":
    cli()

