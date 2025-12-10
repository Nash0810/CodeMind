"""Advanced AST parser using Tree-sitter with full metadata extraction."""

from tree_sitter import Node, Parser
from pathlib import Path
from typing import List
from .language_config import get_language_config
from .data_structures import FileMetadata, FunctionMetadata, ClassMetadata
from .extractors import (
    extract_docstring,
    extract_parameters,
    extract_return_type,
    extract_decorators,
    is_async_function,
    extract_function_calls,
    extract_base_classes,
    compute_content_hash
)


def parse_file(file_path: str) -> FileMetadata:
    """
    Parses a single source file and extracts complete structure.
    
    Args:
        file_path: Path to source file
        
    Returns:
        FileMetadata with extracted functions and classes with full metadata
    """
    path = Path(file_path)
    
    # Get language configuration
    config = get_language_config(path.suffix)
    
    # Read file
    with open(file_path, 'rb') as f:
        code_bytes = f.read()
    
    # Parse with Tree-sitter
    parser = Parser()
    parser.language = config['language']
    tree = parser.parse(code_bytes)
    root = tree.root_node
    
    # Create file metadata
    file_meta = FileMetadata(
        file_path=str(file_path),
        language=config['name'],
        total_lines=len(code_bytes.decode('utf-8', errors='ignore').split('\n'))
    )
    
    # Extract functions and classes recursively, handling decorators
    def extract_from_node(node):
        if node.type == 'decorated_definition':
            # Get the actual definition inside the decorated node
            for child in node.children:
                if child.type in config['function_types']:
                    file_meta.functions.append(extract_complete_function(node, code_bytes, file_path))
                elif child.type in config['class_types']:
                    file_meta.classes.append(extract_complete_class(node, code_bytes, file_path))
                    # Recursively extract methods from the class
                    extract_from_node(child)
        elif node.type in config['function_types']:
            file_meta.functions.append(extract_complete_function(node, code_bytes, file_path))
        elif node.type in config['class_types']:
            file_meta.classes.append(extract_complete_class(node, code_bytes, file_path))
            # Recursively extract methods from the class
            for child in node.children:
                extract_from_node(child)
        else:
            # Recurse into children
            for child in node.children:
                extract_from_node(child)
    
    extract_from_node(root)
    
    file_meta.total_functions = len(file_meta.functions)
    file_meta.total_classes = len(file_meta.classes)
    
    return file_meta


def find_nodes_by_types(node: Node, node_types: List[str]) -> List[Node]:
    """Recursively finds all nodes matching any of the given types"""
    results = []
    
    if node.type in node_types:
        results.append(node)
    
    for child in node.children:
        results.extend(find_nodes_by_types(child, node_types))
    
    return results


def extract_complete_function(node: Node, code_bytes: bytes, file_path: str) -> FunctionMetadata:
    """Extracts complete function metadata including advanced fields"""
    # If passed a decorated_definition, extract the actual function_definition
    actual_node = node
    if node.type == 'decorated_definition':
        for child in node.children:
            if child.type == 'function_definition':
                actual_node = child
                break
    
    # Get function name
    name_node = actual_node.child_by_field_name('name')
    name = name_node.text.decode('utf-8') if name_node else 'anonymous'
    
    # Get code (from the top node, which may include decorators)
    code = code_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
    
    # Get line numbers (from the top node)
    line_start = node.start_point[0] + 1  # Tree-sitter is 0-indexed
    line_end = node.end_point[0] + 1
    
    return FunctionMetadata(
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        code=code,
        docstring=extract_docstring(actual_node, code_bytes),
        parameters=extract_parameters(actual_node, code_bytes),
        return_type=extract_return_type(actual_node, code_bytes),
        decorators=extract_decorators(node, code_bytes),  # Extract from top node (decorated_definition or function_definition)
        is_async=is_async_function(actual_node),
        calls=extract_function_calls(actual_node, code_bytes),
        content_hash=compute_content_hash(code)
    )


def extract_complete_class(node: Node, code_bytes: bytes, file_path: str) -> ClassMetadata:
    """Extracts complete class metadata including advanced fields"""
    # If passed a decorated_definition, extract the actual class_definition
    actual_node = node
    if node.type == 'decorated_definition':
        for child in node.children:
            if child.type == 'class_definition':
                actual_node = child
                break
    
    # Get class name
    name_node = actual_node.child_by_field_name('name')
    name = name_node.text.decode('utf-8') if name_node else 'anonymous'
    
    # Get code (from the top node, which may include decorators)
    code = code_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
    
    # Get line numbers (from the top node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    
    return ClassMetadata(
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        code=code,
        docstring=extract_docstring(actual_node, code_bytes),
        base_classes=extract_base_classes(actual_node, code_bytes),
        decorators=extract_decorators(node, code_bytes)  # Extract from top node
    )
