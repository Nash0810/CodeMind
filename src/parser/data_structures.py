"""Data structures for CodeMind parser."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class FunctionMetadata:
    """Complete representation of a parsed function"""
    name: str
    file_path: str
    line_start: int
    line_end: int
    code: str
    docstring: Optional[str] = None
    parameters: List[Dict[str, Optional[str]]] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    calls: List[str] = field(default_factory=list)
    content_hash: str = ""

@dataclass
class ClassMetadata:
    """Complete representation of a parsed class"""
    name: str
    file_path: str
    line_start: int
    line_end: int
    code: str
    docstring: Optional[str] = None
    base_classes: List[str] = field(default_factory=list)
    methods: List[FunctionMetadata] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)

@dataclass
class FileMetadata:
    """Complete representation of a parsed file"""
    file_path: str
    language: str
    functions: List[FunctionMetadata] = field(default_factory=list)
    classes: List[ClassMetadata] = field(default_factory=list)
    imports: List[Dict[str, Any]] = field(default_factory=list)
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
