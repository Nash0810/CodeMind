"""Directory walking and file discovery for CodeMind parser."""

from pathlib import Path
from typing import List, Iterator
from .language_config import LANGUAGE_REGISTRY
from .ast_parser import parse_file
from .data_structures import FileMetadata
import time


def walk_directory(repo_path: str, exclude_dirs: List[str] = None) -> List[FileMetadata]:
    """
    Recursively walks a directory and parses all supported files.
    
    Args:
        repo_path: Path to repository root
        exclude_dirs: Directory names to skip (e.g., ['venv', 'node_modules'])
        
    Returns:
        List of FileMetadata for all parsed files
    """
    if exclude_dirs is None:
        exclude_dirs = [
            'venv', '.venv', 'env', '.env',
            'node_modules', '__pycache__', '.git',
            'build', 'dist', '.eggs', '.egg-info',
            '.pytest_cache', '.tox'
        ]
    
    results = []
    repo = Path(repo_path)
    
    # Get all supported extensions
    supported_extensions = list(LANGUAGE_REGISTRY.keys())
    
    print(f"Walking directory: {repo_path}")
    print(f"Supported extensions: {supported_extensions}")
    print(f"Excluded directories: {exclude_dirs}\n")
    
    start_time = time.time()
    file_count = 0
    error_count = 0
    
    for file_path in repo.rglob('*'):
        # Skip if in excluded directory
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue
        
        # Check if supported file type
        if file_path.suffix in supported_extensions and file_path.is_file():
            try:
                file_count += 1
                print(f"Parsing [{file_count}]: {file_path.relative_to(repo)}")
                
                parsed = parse_file(str(file_path))
                results.append(parsed)
                
            except Exception as e:
                error_count += 1
                print(f"  ERROR: {e}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Parsing Complete")
    print(f"{'='*60}")
    print(f"Files parsed: {file_count}")
    print(f"Errors: {error_count}")
    print(f"Total time: {elapsed:.2f}s")
    if file_count > 0:
        print(f"Average: {elapsed/file_count:.3f}s per file")
    
    return results


def walk_directory_lazy(repo_path: str, exclude_dirs: List[str] = None) -> Iterator[FileMetadata]:
    """
    Lazy version that yields results one at a time (memory efficient for large repos).
    
    Args:
        repo_path: Path to repository root
        exclude_dirs: Directory names to skip
        
    Yields:
        FileMetadata for each parsed file
    """
    if exclude_dirs is None:
        exclude_dirs = ['venv', '.venv', 'node_modules', '__pycache__', '.git', 'build', 'dist']
    
    repo = Path(repo_path)
    supported_extensions = list(LANGUAGE_REGISTRY.keys())
    
    for file_path in repo.rglob('*'):
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue
        
        if file_path.suffix in supported_extensions and file_path.is_file():
            try:
                yield parse_file(str(file_path))
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
