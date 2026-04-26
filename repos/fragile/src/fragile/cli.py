from pathlib import Path

import click
import flogging
from hydraclick import hydra_command

from fragile.app.montezuma import run_serve_montezuma


CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


@click.group()
def cli():
    """Run command line interface for hydraclick."""
    flogging.setup(allow_trailing_dot=True)


@cli.command()
@hydra_command(config_path=CONFIG_DIR, config_name="montezuma", as_kwargs=True)
def montezuma(**kwargs):
    """Serve the application.

    Args:
        **kwargs: Keyword arguments.

    Returns:
        int: A return code.

    """
    flogging.setup(allow_trailing_dot=True)

    click.echo(f"Serving the application with the following arguments: {kwargs}")
    return run_serve_montezuma(**kwargs)
