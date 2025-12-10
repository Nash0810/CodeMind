"""CodeMind CLI - Command line interface for code analysis."""

import click
import json
from pathlib import Path
from ..parser.directory_walker import walk_directory
from ..parser.profiling import profile_parser


@click.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='code_structure.json', help='Output JSON file')
@click.option('--profile', is_flag=True, help='Enable profiling')
@click.option('--exclude', multiple=True, help='Directories to exclude (can specify multiple times)')
def parse(repo_path: str, output: str, profile: bool, exclude: tuple):
    """
    Parse a repository and extract code structure.
    
    Examples:
        codemind parse /path/to/repo
        codemind parse /path/to/repo --profile
        codemind parse /path/to/repo --exclude venv --exclude tests
    """
    click.echo(f"Parsing repository: {repo_path}")
    
    exclude_list = list(exclude) if exclude else None
    
    if profile:
        # Use profiler
        stats = profile_parser(repo_path)
        click.echo(f"\n✅ Profiling complete. View with: snakeviz parser_profile.prof")
    else:
        # Normal parsing
        results = walk_directory(repo_path, exclude_dirs=exclude_list)
        
        # Convert to JSON-serializable format
        json_results = []
        for file_meta in results:
            json_results.append({
                'file': file_meta.file_path,
                'language': file_meta.language,
                'functions': [
                    {
                        'name': f.name,
                        'line_start': f.line_start,
                        'line_end': f.line_end,
                        'docstring': f.docstring,
                        'parameters': f.parameters,
                        'return_type': f.return_type,
                        'decorators': f.decorators,
                        'is_async': f.is_async,
                        'calls': f.calls,
                        'code': f.code
                    }
                    for f in file_meta.functions
                ],
                'classes': [
                    {
                        'name': c.name,
                        'line_start': c.line_start,
                        'line_end': c.line_end,
                        'docstring': c.docstring,
                        'base_classes': c.base_classes,
                        'methods': [
                            {
                                'name': m.name,
                                'decorators': m.decorators
                            }
                            for m in c.methods
                        ]
                    }
                    for c in file_meta.classes
                ],
                'imports': file_meta.imports
            })
        
        # Save to file
        with open(output, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        total_functions = sum(len(f.functions) for f in results)
        total_classes = sum(len(f.classes) for f in results)
        
        click.echo(f"\n✅ Parsing complete!")
        click.echo(f"Files: {len(results)}")
        click.echo(f"Functions: {total_functions}")
        click.echo(f"Classes: {total_classes}")
        click.echo(f"Output: {output}")


if __name__ == '__main__':
    parse()
