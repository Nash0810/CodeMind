"""
Graph-based context packing for LLM-ready code context assembly.

Intelligently selects code snippets and their dependencies using call graph
traversal to fit within token budgets while maintaining semantic coherence.
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class CodeSnippet:
    """A code snippet with metadata."""
    id: str  # "file:line:name"
    name: str
    file: str
    line: int
    content: str
    type: str  # "function", "class", "method"
    dependencies: List[str] = field(default_factory=list)  # IDs of functions it calls
    dependents: List[str] = field(default_factory=list)    # IDs of functions that call it
    token_count: int = 0
    
    def estimate_tokens(self) -> int:
        """Estimate token count using rough heuristic: ~4 chars per token."""
        if not self.token_count:
            self.token_count = len(self.content) // 4
        return self.token_count


@dataclass
class PackedContext:
    """Assembled context ready for LLM."""
    primary_result: CodeSnippet  # The main search result
    related_code: List[CodeSnippet] = field(default_factory=list)
    dependencies: List[CodeSnippet] = field(default_factory=list)
    dependents: List[CodeSnippet] = field(default_factory=list)
    
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens in packed context."""
        return sum(s.estimate_tokens() for s in self._all_snippets())
    
    def _all_snippets(self) -> List[CodeSnippet]:
        """Get all snippets in context."""
        snippets = [self.primary_result]
        snippets.extend(self.related_code)
        snippets.extend(self.dependencies)
        snippets.extend(self.dependents)
        return snippets
    
    def to_markdown(self) -> str:
        """Format context as markdown for LLM."""
        lines = []
        
        # Primary result
        lines.append(f"# Primary Result: {self.primary_result.name}")
        lines.append(f"**File**: `{self.primary_result.file}:{self.primary_result.line}`")
        lines.append(f"**Type**: {self.primary_result.type}")
        lines.append("")
        lines.append("```python")
        lines.append(self.primary_result.content)
        lines.append("```")
        lines.append("")
        
        # Dependencies (functions this calls)
        if self.dependencies:
            lines.append("## Functions Called (Dependencies)")
            for snippet in self.dependencies:
                lines.append(f"### {snippet.name}")
                lines.append(f"**File**: `{snippet.file}:{snippet.line}`")
                lines.append("```python")
                lines.append(snippet.content)
                lines.append("```")
                lines.append("")
        
        # Dependents (functions that call this)
        if self.dependents:
            lines.append("## Called By (Dependents)")
            for snippet in self.dependents:
                lines.append(f"### {snippet.name}")
                lines.append(f"**File**: `{snippet.file}:{snippet.line}`")
                lines.append("```python")
                lines.append(snippet.content)
                lines.append("```")
                lines.append("")
        
        # Related code
        if self.related_code:
            lines.append("## Related Code")
            for snippet in self.related_code:
                lines.append(f"### {snippet.name}")
                lines.append(f"**File**: `{snippet.file}:{snippet.line}`")
                lines.append("```python")
                lines.append(snippet.content)
                lines.append("```")
                lines.append("")
        
        return "\n".join(lines)
    
    def to_text(self) -> str:
        """Format context as plain text."""
        lines = []
        
        # Primary result
        lines.append(f"PRIMARY RESULT: {self.primary_result.name}")
        lines.append(f"File: {self.primary_result.file}:{self.primary_result.line}")
        lines.append(f"Type: {self.primary_result.type}")
        lines.append("-" * 60)
        lines.append(self.primary_result.content)
        lines.append("")
        
        # Dependencies
        if self.dependencies:
            lines.append("DEPENDENCIES (Functions Called):")
            lines.append("-" * 60)
            for snippet in self.dependencies:
                lines.append(f"\n{snippet.name} ({snippet.file}:{snippet.line})")
                lines.append(snippet.content)
            lines.append("")
        
        # Dependents
        if self.dependents:
            lines.append("DEPENDENTS (Called By):")
            lines.append("-" * 60)
            for snippet in self.dependents:
                lines.append(f"\n{snippet.name} ({snippet.file}:{snippet.line})")
                lines.append(snippet.content)
            lines.append("")
        
        # Related
        if self.related_code:
            lines.append("RELATED CODE:")
            lines.append("-" * 60)
            for snippet in self.related_code:
                lines.append(f"\n{snippet.name} ({snippet.file}:{snippet.line})")
                lines.append(snippet.content)
        
        return "\n".join(lines)


class ContextPacker:
    """
    Assembles code context for LLM using call graph relationships.
    
    Strategy:
    1. Start with primary search result
    2. Use BFS on call graph to include dependencies
    3. Respect token budget by selecting most relevant snippets
    4. Format into markdown for LLM consumption
    """
    
    def __init__(self, token_budget: int = 8000):
        """
        Initialize context packer.
        
        Args:
            token_budget: Maximum tokens to include in context (~8000 for 8k context)
        """
        self.token_budget = token_budget
        self.snippet_cache: Dict[str, CodeSnippet] = {}
        self.call_graph: Dict[str, List[str]] = {}  # Maps ID -> IDs it calls
    
    def add_snippet(self, snippet: CodeSnippet) -> None:
        """Register a code snippet."""
        self.snippet_cache[snippet.id] = snippet
    
    def add_call_relationship(self, caller_id: str, callee_id: str) -> None:
        """Register that caller_id calls callee_id."""
        if caller_id not in self.call_graph:
            self.call_graph[caller_id] = []
        self.call_graph[caller_id].append(callee_id)
    
    def pack_context(
        self,
        primary_result_id: str,
        max_depth: int = 2,
        include_dependents: bool = True,
        include_related: bool = True
    ) -> Optional[PackedContext]:
        """
        Pack context around a primary search result.
        
        Args:
            primary_result_id: The ID of the main search result
            max_depth: Maximum BFS depth for dependency traversal
            include_dependents: Whether to include functions that call this one
            include_related: Whether to include semantically related code
            
        Returns:
            PackedContext with assembled code snippets, or None if not found
        """
        if primary_result_id not in self.snippet_cache:
            return None
        
        primary = self.snippet_cache[primary_result_id]
        context = PackedContext(primary_result=primary)
        
        # BFS to find dependencies (functions this calls)
        dependencies = self._find_dependencies(primary_result_id, max_depth)
        context.dependencies = [self.snippet_cache[dep_id] for dep_id in dependencies 
                                if dep_id in self.snippet_cache]
        
        # Find dependents (functions that call this)
        if include_dependents:
            dependents = self._find_dependents(primary_result_id, max_depth)
            context.dependents = [self.snippet_cache[dep_id] for dep_id in dependents
                                 if dep_id in self.snippet_cache]
        
        # Trim to fit token budget
        self._trim_to_budget(context)
        
        return context
    
    def _find_dependencies(self, node_id: str, max_depth: int) -> Set[str]:
        """
        BFS to find all functions called by node_id.
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth
            
        Returns:
            Set of function IDs called (directly or indirectly)
        """
        visited = set()
        queue = deque([(node_id, 0)])  # (node_id, depth)
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Add all functions called by current
            for callee_id in self.call_graph.get(current_id, []):
                if callee_id not in visited:
                    queue.append((callee_id, depth + 1))
        
        # Remove the starting node
        visited.discard(node_id)
        return visited
    
    def _find_dependents(self, node_id: str, max_depth: int) -> Set[str]:
        """
        BFS to find all functions that call node_id.
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth
            
        Returns:
            Set of function IDs that call this node
        """
        # Build reverse graph (callee -> callers)
        reverse_graph: Dict[str, List[str]] = {}
        for caller_id, callees in self.call_graph.items():
            for callee_id in callees:
                if callee_id not in reverse_graph:
                    reverse_graph[callee_id] = []
                reverse_graph[callee_id].append(caller_id)
        
        # BFS from node_id backwards
        visited = set()
        queue = deque([(node_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Add all functions that call current
            for caller_id in reverse_graph.get(current_id, []):
                if caller_id not in visited:
                    queue.append((caller_id, depth + 1))
        
        # Remove the starting node
        visited.discard(node_id)
        return visited
    
    def _trim_to_budget(self, context: PackedContext) -> None:
        """
        Trim context snippets to fit within token budget.
        
        Priority: Primary > Dependencies > Dependents > Related
        """
        current_tokens = context.primary_result.estimate_tokens()
        remaining = self.token_budget - current_tokens
        
        # Add dependencies (highest priority after primary)
        deps_to_keep = []
        for snippet in context.dependencies:
            tokens = snippet.estimate_tokens()
            if tokens <= remaining:
                deps_to_keep.append(snippet)
                remaining -= tokens
        context.dependencies = deps_to_keep
        
        # Add dependents
        dependents_to_keep = []
        for snippet in context.dependents:
            tokens = snippet.estimate_tokens()
            if tokens <= remaining:
                dependents_to_keep.append(snippet)
                remaining -= tokens
        context.dependents = dependents_to_keep
        
        # Add related (lowest priority)
        related_to_keep = []
        for snippet in context.related_code:
            tokens = snippet.estimate_tokens()
            if tokens <= remaining:
                related_to_keep.append(snippet)
                remaining -= tokens
        context.related_code = related_to_keep
    
    def pack_multiple_results(
        self,
        result_ids: List[str],
        max_depth: int = 1,
        distribute_budget: bool = True
    ) -> List[PackedContext]:
        """
        Pack context for multiple search results.
        
        Args:
            result_ids: List of result IDs to pack
            max_depth: Maximum BFS depth
            distribute_budget: If True, distribute token budget across results
            
        Returns:
            List of PackedContext objects
        """
        if distribute_budget:
            # Distribute token budget evenly
            per_result_budget = self.token_budget // len(result_ids)
            original_budget = self.token_budget
            self.token_budget = per_result_budget
        
        contexts = []
        for result_id in result_ids:
            context = self.pack_context(result_id, max_depth)
            if context:
                contexts.append(context)
        
        if distribute_budget:
            self.token_budget = original_budget
        
        return contexts
    
    def get_statistics(self, context: PackedContext) -> Dict[str, int]:
        """Get statistics about packed context."""
        return {
            'primary_tokens': context.primary_result.estimate_tokens(),
            'dependencies_count': len(context.dependencies),
            'dependencies_tokens': sum(s.estimate_tokens() for s in context.dependencies),
            'dependents_count': len(context.dependents),
            'dependents_tokens': sum(s.estimate_tokens() for s in context.dependents),
            'related_count': len(context.related_code),
            'related_tokens': sum(s.estimate_tokens() for s in context.related_code),
            'total_tokens': context.total_tokens,
            'snippets_count': len(context._all_snippets()),
        }
