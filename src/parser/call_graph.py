"""Call graph construction for CodeMind parser."""

from typing import Dict, Set, List
from collections import deque
from .data_structures import FunctionMetadata, FileMetadata


class CallGraph:
    """
    Directed graph representing function call relationships.
    
    Nodes: Functions (keyed by "file_path:function_name")
    Edges: Function A → Function B means "A calls B"
    """
    
    def __init__(self):
        # Forward edges: caller → callees
        self.calls: Dict[str, Set[str]] = {}
        
        # Reverse edges: callee → callers
        self.called_by: Dict[str, Set[str]] = {}
        
        # Function metadata lookup
        self.function_map: Dict[str, FunctionMetadata] = {}
    
    def add_function(self, func: FunctionMetadata):
        """Register a function in the graph"""
        key = self._make_key(func.file_path, func.name)
        
        self.function_map[key] = func
        
        # Initialize edges
        if key not in self.calls:
            self.calls[key] = set()
        if key not in self.called_by:
            self.called_by[key] = set()
    
    def add_call(self, caller_file: str, caller_name: str, callee_name: str):
        """Add a call edge: caller → callee"""
        caller_key = self._make_key(caller_file, caller_name)
        
        # Try to find the callee in our function map
        # This is fuzzy because we only have the function name, not the full path
        callee_key = self._find_function_key(callee_name)
        
        if callee_key and caller_key in self.calls:
            self.calls[caller_key].add(callee_key)
            
            if callee_key not in self.called_by:
                self.called_by[callee_key] = set()
            self.called_by[callee_key].add(caller_key)
    
    def build_from_files(self, files: List[FileMetadata]):
        """Build call graph from parsed files"""
        print("Building call graph...")
        
        # First pass: register all functions
        for file_meta in files:
            for func in file_meta.functions:
                self.add_function(func)
            
            for cls in file_meta.classes:
                for method in cls.methods:
                    self.add_function(method)
        
        print(f"  Registered {len(self.function_map)} functions")
        
        # Second pass: build edges
        edge_count = 0
        for file_meta in files:
            for func in file_meta.functions:
                for callee in func.calls:
                    self.add_call(file_meta.file_path, func.name, callee)
                    edge_count += 1
            
            for cls in file_meta.classes:
                for method in cls.methods:
                    for callee in method.calls:
                        self.add_call(file_meta.file_path, method.name, callee)
                        edge_count += 1
        
        print(f"  Built {edge_count} call edges")
    
    def get_dependencies(self, func_key: str, max_depth: int = 2) -> List[str]:
        """
        Get transitive dependencies using BFS.
        
        Args:
            func_key: Function identifier (file:name)
            max_depth: Maximum traversal depth
            
        Returns:
            List of function keys this function depends on
        """
        if func_key not in self.calls:
            return []
        
        visited = set()
        queue = deque([(func_key, 0)])
        dependencies = []
        
        while queue:
            current, depth = queue.popleft()
            
            if current in visited or depth > max_depth:
                continue
            
            visited.add(current)
            if current != func_key:  # Don't include the starting function
                dependencies.append(current)
            
            # Add functions this one calls
            for callee in self.calls.get(current, []):
                if callee not in visited:
                    queue.append((callee, depth + 1))
        
        return dependencies
    
    def get_dependents(self, func_key: str, max_depth: int = 1) -> List[str]:
        """
        Get functions that depend on this function (reverse direction).
        
        Args:
            func_key: Function identifier
            max_depth: Maximum traversal depth
            
        Returns:
            List of function keys that call this function
        """
        if func_key not in self.called_by:
            return []
        
        visited = set()
        queue = deque([(func_key, 0)])
        dependents = []
        
        while queue:
            current, depth = queue.popleft()
            
            if current in visited or depth > max_depth:
                continue
            
            visited.add(current)
            if current != func_key:
                dependents.append(current)
            
            # Add functions that call this one
            for caller in self.called_by.get(current, []):
                if caller not in visited:
                    queue.append((caller, depth + 1))
        
        return dependents
    
    def get_function(self, func_key: str) -> FunctionMetadata:
        """Get function metadata by key"""
        return self.function_map.get(func_key)
    
    def _make_key(self, file_path: str, func_name: str) -> str:
        """Create unique key for a function"""
        return f"{file_path}:{func_name}"
    
    def _find_function_key(self, func_name: str) -> str:
        """
        Fuzzy search for function by name.
        Returns the first match or None.
        """
        for key in self.function_map.keys():
            if key.endswith(f":{func_name}"):
                return key
        return None
    
    def export_to_json(self) -> dict:
        """Export graph to JSON format"""
        return {
            'nodes': [
                {
                    'id': key,
                    'name': func.name,
                    'file': func.file_path,
                    'lines': f"{func.line_start}-{func.line_end}"
                }
                for key, func in self.function_map.items()
            ],
            'edges': [
                {'from': caller, 'to': callee}
                for caller, callees in self.calls.items()
                for callee in callees
            ]
        }
