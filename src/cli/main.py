"""Main CLI entry point for CodeMind."""

import click
from .parse import parse
from .query import search_command
from .tui import run_tui


@click.group()
def cli():
    """CodeMind - Intelligent Code Analysis Tool"""
    pass


cli.add_command(parse)
cli.add_command(search_command, name='search')


@cli.command(name='interactive')
def interactive_command():
    """Launch interactive TUI interface."""
    run_tui()



if __name__ == '__main__':
    cli()
