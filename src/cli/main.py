"""Main CLI entry point for CodeMind."""

import click
from .parse import parse


@click.group()
def cli():
    """CodeMind - Intelligent Code Analysis Tool"""
    pass


cli.add_command(parse)


if __name__ == '__main__':
    cli()
