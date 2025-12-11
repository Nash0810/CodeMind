"""
Advanced filtering and sorting for search results.

Provides sophisticated result filtering and sorting capabilities.
"""

from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import re


class SortOrder(Enum):
    """Result sorting options."""
    RELEVANCE = "relevance"  # Combined score (default)
    NAME = "name"            # Alphabetical by name
    FILE = "file"            # By file path
    TYPE = "type"            # By function/class
    LATENCY = "latency"      # By computation time (reverse)


class FilterMode(Enum):
    """Filter operation modes."""
    AND = "and"  # All conditions must match
    OR = "or"    # Any condition can match


class ResultFilter:
    """Advanced filtering for search results."""
    
    def __init__(self):
        """Initialize filter."""
        self.filters: List[Callable] = []
        self.mode = FilterMode.AND
    
    def by_type(self, result_type: str) -> 'ResultFilter':
        """Filter by function or class type.
        
        Args:
            result_type: 'function' or 'class'
            
        Returns:
            Self for chaining
        """
        def filter_func(result):
            metadata = result.metadata if hasattr(result, 'metadata') else result
            return metadata.get('type', '').lower() == result_type.lower()
        
        self.filters.append(filter_func)
        return self
    
    def by_file(self, filename_pattern: str, exact: bool = False) -> 'ResultFilter':
        """Filter by filename or path pattern.
        
        Args:
            filename_pattern: Pattern to match (supports wildcards and regex)
            exact: If True, match entire path; if False, match any part
            
        Returns:
            Self for chaining
        """
        def filter_func(result):
            metadata = result.metadata if hasattr(result, 'metadata') else result
            filepath = metadata.get('file', '').lower()
            pattern = filename_pattern.lower()
            
            if exact:
                return filepath == pattern
            else:
                return pattern in filepath or filepath.endswith(pattern)
        
        self.filters.append(filter_func)
        return self
    
    def by_docstring(self, search_term: str) -> 'ResultFilter':
        """Filter by docstring content.
        
        Args:
            search_term: Term to search in docstrings
            
        Returns:
            Self for chaining
        """
        def filter_func(result):
            metadata = result.metadata if hasattr(result, 'metadata') else result
            docstring = metadata.get('docstring', '').lower()
            return search_term.lower() in docstring
        
        self.filters.append(filter_func)
        return self
    
    def by_score_range(self, min_score: float = 0.0, max_score: float = 1.0) -> 'ResultFilter':
        """Filter by relevance score range.
        
        Args:
            min_score: Minimum score (0-1)
            max_score: Maximum score (0-1)
            
        Returns:
            Self for chaining
        """
        def filter_func(result):
            score = result.combined_score if hasattr(result, 'combined_score') else 0.0
            return min_score <= score <= max_score
        
        self.filters.append(filter_func)
        return self
    
    def by_code_pattern(self, pattern: str) -> 'ResultFilter':
        """Filter by code pattern (regex).
        
        Args:
            pattern: Regex pattern to match in code
            
        Returns:
            Self for chaining
        """
        def filter_func(result):
            metadata = result.metadata if hasattr(result, 'metadata') else result
            code = metadata.get('code', '')
            try:
                return re.search(pattern, code, re.IGNORECASE) is not None
            except:
                return False
        
        self.filters.append(filter_func)
        return self
    
    def by_name_pattern(self, pattern: str, regex: bool = False) -> 'ResultFilter':
        """Filter by name pattern.
        
        Args:
            pattern: Pattern to match (string or regex)
            regex: If True, treat pattern as regex
            
        Returns:
            Self for chaining
        """
        def filter_func(result):
            metadata = result.metadata if hasattr(result, 'metadata') else result
            name = metadata.get('name', '').lower()
            
            if regex:
                try:
                    return re.search(pattern, name, re.IGNORECASE) is not None
                except:
                    return False
            else:
                return pattern.lower() in name
        
        self.filters.append(filter_func)
        return self
    
    def apply(self, results: List) -> List:
        """Apply all filters to results.
        
        Args:
            results: Search results to filter
            
        Returns:
            Filtered results
        """
        if not self.filters:
            return results
        
        filtered = []
        
        for result in results:
            if self.mode == FilterMode.AND:
                # All filters must pass
                if all(f(result) for f in self.filters):
                    filtered.append(result)
            else:  # OR mode
                # Any filter can pass
                if any(f(result) for f in self.filters):
                    filtered.append(result)
        
        return filtered
    
    def reset(self) -> 'ResultFilter':
        """Reset all filters.
        
        Returns:
            Self for chaining
        """
        self.filters.clear()
        return self


class ResultSorter:
    """Sorting for search results."""
    
    @staticmethod
    def sort(results: List, 
             order: SortOrder = SortOrder.RELEVANCE,
             reverse: bool = False) -> List:
        """Sort results by specified order.
        
        Args:
            results: Results to sort
            order: Sort order (relevance, name, file, type)
            reverse: Reverse sort order
            
        Returns:
            Sorted results
        """
        if order == SortOrder.RELEVANCE:
            return sorted(
                results,
                key=lambda r: r.combined_score if hasattr(r, 'combined_score') else 0.0,
                reverse=True
            )
        
        elif order == SortOrder.NAME:
            return sorted(
                results,
                key=lambda r: (r.metadata if hasattr(r, 'metadata') else r).get('name', '').lower(),
                reverse=reverse
            )
        
        elif order == SortOrder.FILE:
            return sorted(
                results,
                key=lambda r: (r.metadata if hasattr(r, 'metadata') else r).get('file', '').lower(),
                reverse=reverse
            )
        
        elif order == SortOrder.TYPE:
            return sorted(
                results,
                key=lambda r: (r.metadata if hasattr(r, 'metadata') else r).get('type', '').lower(),
                reverse=reverse
            )
        
        else:
            return results
    
    @staticmethod
    def multi_sort(results: List,
                   orders: List[tuple] = None) -> List:
        """Sort by multiple criteria.
        
        Args:
            results: Results to sort
            orders: List of (SortOrder, reverse) tuples
            
        Returns:
            Sorted results
        """
        if not orders:
            orders = [(SortOrder.RELEVANCE, False)]
        
        sorted_results = results
        
        # Apply sorts in reverse order (last sort has highest priority)
        for sort_order, reverse in reversed(orders):
            sorted_results = ResultSorter.sort(sorted_results, sort_order, reverse)
        
        return sorted_results
    
    @staticmethod
    def group_by(results: List, 
                 group_key: str = 'type') -> Dict[str, List]:
        """Group results by a field.
        
        Args:
            results: Results to group
            group_key: Field to group by ('type', 'file', 'name')
            
        Returns:
            Dictionary of grouped results
        """
        groups: Dict[str, List] = {}
        
        for result in results:
            metadata = result.metadata if hasattr(result, 'metadata') else result
            key = metadata.get(group_key, 'unknown')
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append(result)
        
        return groups
