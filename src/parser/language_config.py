"""Language configuration for CodeMind parser."""

from tree_sitter import Language
import tree_sitter_python
from typing import Dict, Any

# Language configuration
PYTHON_CONFIG = {
    'name': 'python',
    'extensions': ['.py'],
    'language': Language(tree_sitter_python.language()),
    'function_types': ['function_definition'],
    'class_types': ['class_definition'],
    'has_docstrings': True,
}

# Registry of supported languages
LANGUAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    '.py': PYTHON_CONFIG,
}

def get_language_config(file_extension: str) -> Dict[str, Any]:
    """Get language configuration for a file extension"""
    if file_extension not in LANGUAGE_REGISTRY:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    return LANGUAGE_REGISTRY[file_extension]
