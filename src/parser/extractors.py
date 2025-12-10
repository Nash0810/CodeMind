"""Advanced extraction functions for AST parsing."""

from tree_sitter import Node
from typing import List, Dict, Optional
import hashlib
import re


def extract_docstring(node: Node, code_bytes: bytes) -> Optional[str]:
    """
    Extracts docstring from a function or class.
    In Python, docstring is the first statement if it's a string literal.
    """
    body = node.child_by_field_name('body')
    if not body or not body.children:
        return None
    
    # First child of body should be the docstring
    for child in body.children:
        if child.type == 'expression_statement':
            string_node = child.children[0] if child.children else None
            if string_node and string_node.type == 'string':
                # Extract text and clean quotes
                doc_text = code_bytes[string_node.start_byte:string_node.end_byte].decode('utf-8', errors='ignore')
                # Remove quotes and leading/trailing whitespace
                doc_text = doc_text.strip('"""').strip("'''").strip('"').strip("'").strip()
                return doc_text if doc_text else None
        # Docstring is always first non-comment statement
        elif child.type != 'comment':
            break
    
    return None


def extract_parameters(node: Node, code_bytes: bytes) -> List[Dict[str, Optional[str]]]:
    """
    Extracts function parameters with type hints.
    
    Returns:
        List of dicts with 'name' and 'type' keys
    """
    params = []
    params_node = node.child_by_field_name('parameters')
    
    if not params_node:
        return params
    
    for child in params_node.children:
        param_name = None
        param_type = None
        
        if child.type == 'identifier':
            # Simple parameter without type hint
            param_name = child.text.decode('utf-8')
        
        elif child.type == 'typed_parameter':
            # Parameter with type hint: name: type
            for subchild in child.children:
                if subchild.type == 'identifier':
                    param_name = subchild.text.decode('utf-8')
                elif subchild.type == 'type':
                    param_type = code_bytes[subchild.start_byte:subchild.end_byte].decode('utf-8', errors='ignore')
        
        elif child.type == 'typed_default_parameter':
            # Parameter with type hint and default value: name: type = value
            for subchild in child.children:
                if subchild.type == 'identifier':
                    param_name = subchild.text.decode('utf-8')
                elif subchild.type == 'type':
                    param_type = code_bytes[subchild.start_byte:subchild.end_byte].decode('utf-8', errors='ignore')
        
        elif child.type == 'default_parameter':
            # Parameter with default value (no type): name = value
            for subchild in child.children:
                if subchild.type == 'identifier':
                    param_name = subchild.text.decode('utf-8')
        
        if param_name:
            params.append({
                'name': param_name,
                'type': param_type
            })
    
    # Filter out 'self' and 'cls'
    params = [p for p in params if p['name'] not in ['self', 'cls']]
    
    return params


def extract_return_type(node: Node, code_bytes: bytes) -> Optional[str]:
    """Extracts return type annotation"""
    return_type_node = node.child_by_field_name('return_type')
    
    if return_type_node:
        return code_bytes[return_type_node.start_byte:return_type_node.end_byte].decode('utf-8', errors='ignore')
    
    return None


def extract_decorators(node: Node, code_bytes: bytes) -> List[str]:
    """
    Extracts decorators from function or class.
    """
    decorators = []
    
    # Tree-sitter puts decorators as children of the function_definition node
    for child in node.children:
        if child.type == 'decorator':
            dec_text = code_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='ignore')
            decorators.append(dec_text.strip())
    
    return decorators


def is_async_function(node: Node) -> bool:
    """Checks if function is async"""
    # In Python AST, async functions have an 'async' keyword as a child
    for child in node.children:
        if child.type == 'async':
            return True
    return False


def extract_function_calls(node: Node, code_bytes: bytes) -> List[str]:
    """
    Extracts all function calls within this code block.
    This builds the call graph.
    """
    calls = set()
    
    def walk_for_calls(n: Node):
        if n.type == 'call':
            # Get the function being called
            func_node = n.child_by_field_name('function')
            if func_node:
                call_text = code_bytes[func_node.start_byte:func_node.end_byte].decode('utf-8', errors='ignore')
                # Extract just the function name (not the full path)
                # e.g., "db.query.filter" -> "filter"
                call_name = call_text.split('.')[-1] if '.' in call_text else call_text
                calls.add(call_name)
        
        for child in n.children:
            walk_for_calls(child)
    
    walk_for_calls(node)
    return list(calls)


def extract_base_classes(node: Node, code_bytes: bytes) -> List[str]:
    """Extracts base classes from a class definition"""
    bases = []
    
    # Look for argument_list which contains base classes
    for child in node.children:
        if child.type == 'argument_list':
            for arg in child.children:
                if arg.type == 'identifier':
                    bases.append(arg.text.decode('utf-8'))
                elif arg.type == 'attribute':
                    # Handle dotted names like module.ClassName
                    text = code_bytes[arg.start_byte:arg.end_byte].decode('utf-8', errors='ignore')
                    bases.append(text)
    
    return bases


def compute_content_hash(code: str) -> str:
    """Computes SHA256 hash of code for deduplication"""
    return hashlib.sha256(code.encode('utf-8')).hexdigest()
