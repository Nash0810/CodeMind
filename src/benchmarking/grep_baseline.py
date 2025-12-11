"""
Grep-based code search baseline for benchmarking.

This module wraps grep functionality to provide a baseline comparison
for CodeMind's vector and keyword search performance.
"""

import os
import re
import subprocess
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class GrepResult:
    """Result from grep search."""
    file: str
    line_number: int
    line_content: str
    score: float = 0.0


class GrepBaseline:
    """Grep-based search for baseline comparison."""

    def __init__(self, root_dir: str = "tests/fixtures"):
        """Initialize grep baseline.
        
        Args:
            root_dir: Root directory to search in
        """
        self.root_dir = root_dir
        self.files = self._collect_files()
        self.index = self._build_index()

    def _collect_files(self) -> List[str]:
        """Collect all Python files in the root directory."""
        files = []
        for root, dirs, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('.py'):
                    files.append(os.path.join(root, filename))
        return files

    def _build_index(self) -> Dict[str, List[Tuple[int, str]]]:
        """Build in-memory index of all lines in all files.
        
        Returns:
            Dict mapping filename to list of (line_number, content) tuples
        """
        index = {}
        for file_path in self.files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    index[file_path] = [(i + 1, line.rstrip()) for i, line in enumerate(lines)]
            except Exception as e:
                print(f"Warning: Could not index {file_path}: {e}")
        return index

    def search(self, query: str, max_results: int = 20) -> List[GrepResult]:
        """Search using grep pattern matching.
        
        This simulates grep by doing line-by-line pattern matching
        on all indexed files.
        
        Args:
            query: Search query (treated as regex pattern)
            max_results: Maximum number of results to return
            
        Returns:
            List of GrepResult objects, ordered by frequency
        """
        results = []
        query_lower = query.lower()
        pattern = None
        
        # Try to compile as regex, fallback to literal match
        try:
            pattern = re.compile(query_lower, re.IGNORECASE)
        except re.error:
            # Treat as literal string search
            pass

        # Search through all indexed lines
        match_counts = {}
        for file_path, lines in self.index.items():
            for line_num, content in lines:
                is_match = False
                
                if pattern:
                    is_match = pattern.search(content.lower()) is not None
                else:
                    is_match = query_lower in content.lower()
                
                if is_match:
                    result_key = (file_path, line_num)
                    match_counts[result_key] = match_counts.get(result_key, 0) + 1
                    results.append(GrepResult(
                        file=file_path,
                        line_number=line_num,
                        line_content=content,
                        score=1.0  # All matches have equal score in grep
                    ))

        # Sort by line number (grep typically outputs in file order)
        results.sort(key=lambda r: (r.file, r.line_number))
        
        return results[:max_results]

    def search_function_names(self, query: str, max_results: int = 20) -> List[GrepResult]:
        """Search for function/class definitions matching the query.
        
        This is a specialized search that looks for 'def' and 'class' patterns
        and matches based on keywords in the function name or docstrings.
        
        Args:
            query: Name pattern to search for (keywords will be extracted)
            max_results: Maximum number of results
            
        Returns:
            List of results containing definitions
        """
        results = []
        
        # Extract keywords from query (split on whitespace)
        keywords = [k.lower() for k in query.split()]
        
        for file_path, lines in self.index.items():
            for line_num, content in lines:
                # Look for function or class definitions
                if re.search(r'^\s*(def|class)\s+', content):
                    # Check if any keyword matches the function/class name
                    name_match = re.search(r'(def|class)\s+(\w+)', content)
                    if name_match:
                        func_name = name_match.group(2).lower()
                        
                        # Match if any keyword is in the function name
                        for keyword in keywords:
                            if keyword in func_name:
                                results.append(GrepResult(
                                    file=file_path,
                                    line_number=line_num,
                                    line_content=content,
                                    score=1.0
                                ))
                                break

        # Sort by file and line number
        results.sort(key=lambda r: (r.file, r.line_number))
        
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for r in results:
            key = (r.file, r.line_number)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        
        return deduped[:max_results]


def benchmark_grep_search(queries: List[str], root_dir: str = "tests/fixtures") -> Dict:
    """Benchmark grep-based search.
    
    Args:
        queries: List of search queries
        root_dir: Root directory to search in
        
    Returns:
        Dictionary with timing results
    """
    grep = GrepBaseline(root_dir)
    
    results = {
        'setup_time': 0,
        'queries': []
    }
    
    # Record index build time
    start = time.time()
    grep_copy = GrepBaseline(root_dir)
    results['setup_time'] = time.time() - start
    
    # Run each query
    for query in queries:
        query_start = time.time()
        search_results = grep.search(query)
        query_time = time.time() - query_start
        
        results['queries'].append({
            'query': query,
            'time': query_time,
            'count': len(search_results),
            'results': search_results[:5]  # Keep only first 5 for inspection
        })
    
    return results
