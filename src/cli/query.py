"""
Query command for CodeMind CLI.

Provides semantic and keyword search capabilities with result formatting.
"""

import click
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from src.indexing.vector_store import VectorStore
from src.indexing.keyword_search import KeywordSearch
from src.search.hybrid_search import HybridSearch
from src.query_cache import QueryCache
from src.query_history import QueryHistory
from src.result_filter import ResultFilter, ResultSorter, SortOrder


class QueryEngine:
    """Manages search and result display."""
    
    def __init__(self, persist_dir: str = "./chroma_db", cache_dir: str = ".codemind_cache"):
        """Initialize query engine.
        
        Args:
            persist_dir: Directory for ChromaDB storage
            cache_dir: Directory for query caching
        """
        self.persist_dir = persist_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize search components
        self.vector_store = VectorStore(persist_directory=persist_dir)
        self.keyword_search = KeywordSearch()
        self.hybrid_search = HybridSearch(self.vector_store, self.keyword_search)
        
        # Initialize query cache
        self.query_cache = QueryCache(max_size=100, cache_dir=str(self.cache_dir / "queries"))
        
        # Initialize query history
        self.query_history = QueryHistory(history_file=str(self.cache_dir / "history.json"))
        
        self.console = Console()
        self._indexed = False
    
    def ensure_indexed(self, code_json: str = "code_structure.json"):
        """Ensure code is indexed.
        
        Args:
            code_json: Path to code_structure.json
        """
        if not Path(code_json).exists():
            self.console.print(f"[red]Error:[/red] {code_json} not found", style="bold")
            return False
        
        if self.vector_store.count() == 0:
            self.console.print("[cyan]Indexing code...[/cyan]")
            try:
                with open(code_json) as f:
                    code_data = json.load(f)
                
                self.vector_store.index_code_blocks(code_data)
                self.keyword_search.index_code_blocks(code_data)
                self.console.print(f"[green]Indexed {self.vector_store.count()} code blocks[/green]")
            except Exception as e:
                self.console.print(f"[red]Indexing failed: {e}[/red]")
                return False
        
        self._indexed = True
        return True
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               vector_weight: float = 0.6,
               keyword_weight: float = 0.4) -> Tuple[List[Dict], bool, float]:
        """Execute hybrid search with caching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            vector_weight: Weight for vector search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            
        Returns:
            Tuple of (results, from_cache, latency_seconds)
        """
        if not self._indexed:
            raise RuntimeError("Code not indexed. Call ensure_indexed() first")
        
        # Check cache first
        cache_result = self.query_cache.get(query, vector_weight, keyword_weight, top_k)
        if cache_result:
            results, cache_age = cache_result
            return (results, True, 0.0)  # from_cache=True, latency=0 (served from cache)
        
        # Execute search
        start = time.time()
        results = self.hybrid_search.search(
            query,
            top_k=top_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
        )
        latency = time.time() - start
        
        # Cache results
        self.query_cache.set(query, results, vector_weight, keyword_weight, top_k)
        
        return (results, False, latency)
    
    def filter_results(self, results: List[Dict], 
                      result_type: Optional[str] = None) -> List[Dict]:
        """Filter results by type.
        
        Args:
            results: Search results
            result_type: Filter by 'function' or 'class' (None = all)
            
        Returns:
            Filtered results
        """
        if not result_type:
            return results
        
        return [r for r in results 
                if hasattr(r, 'metadata') and r.metadata.get('type') == result_type]
    
    def advanced_filter(self, results: List[Dict], 
                       result_type: Optional[str] = None,
                       filename: Optional[str] = None,
                       docstring_term: Optional[str] = None,
                       min_score: Optional[float] = None,
                       max_score: Optional[float] = None,
                       code_pattern: Optional[str] = None) -> List[Dict]:
        """Apply advanced filters to results.
        
        Args:
            results: Search results
            result_type: Filter by 'function' or 'class'
            filename: Filter by filename pattern
            docstring_term: Filter by docstring content
            min_score: Minimum relevance score (0-1)
            max_score: Maximum relevance score (0-1)
            code_pattern: Filter by code regex pattern
            
        Returns:
            Filtered results
        """
        filter_obj = ResultFilter()
        
        if result_type:
            filter_obj.by_type(result_type)
        if filename:
            filter_obj.by_file(filename)
        if docstring_term:
            filter_obj.by_docstring(docstring_term)
        if min_score is not None or max_score is not None:
            min_s = min_score or 0.0
            max_s = max_score or 1.0
            filter_obj.by_score_range(min_s, max_s)
        if code_pattern:
            filter_obj.by_code_pattern(code_pattern)
        
        return filter_obj.apply(results)
    
    def sort_results(self, results: List[Dict],
                    sort_by: str = "relevance",
                    reverse: bool = False) -> List[Dict]:
        """Sort results by specified criteria.
        
        Args:
            results: Search results
            sort_by: Sort criteria ('relevance', 'name', 'file', 'type')
            reverse: Reverse sort order
            
        Returns:
            Sorted results
        """
        try:
            sort_order = SortOrder(sort_by.lower())
        except ValueError:
            sort_order = SortOrder.RELEVANCE
        
        return ResultSorter.sort(results, sort_order, reverse)
    
    def group_results(self, results: List[Dict],
                     group_by: str = "type") -> Dict[str, List[Dict]]:
        """Group results by field.
        
        Args:
            results: Search results
            group_by: Field to group by ('type', 'file', 'name')
            
        Returns:
            Dictionary of grouped results
        """
        return ResultSorter.group_by(results, group_by)
    
    def display_results(self, results: List[Dict], 
                       query: str, 
                       show_code: bool = False,
                       show_latency: bool = True):
        """Display search results in formatted table.
        
        Args:
            results: Search results
            query: Original query
            show_code: Include code snippets
            show_latency: Include timing info
        """
        if not results:
            self.console.print(f"[yellow]No results found for: '{query}'[/yellow]")
            return
        
        # Create results table
        table = Table(title=f"Search Results for '{query}'")
        table.add_column("Rank", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("File", style="blue")
        table.add_column("Score", style="yellow")
        
        for i, result in enumerate(results, 1):
            metadata = result.metadata if hasattr(result, 'metadata') else result
            
            name = metadata.get('name', 'Unknown')
            result_type = metadata.get('type', 'unknown')
            file_path = metadata.get('file', 'Unknown')
            score = result.combined_score if hasattr(result, 'combined_score') else 0
            
            # Shorten file path
            file_display = Path(file_path).name if file_path else 'Unknown'
            
            table.add_row(
                str(i),
                name,
                result_type,
                file_display,
                f"{score:.3f}"
            )
        
        self.console.print(table)
        
        # Show code snippets if requested
        if show_code and results:
            self.console.print("\n[bold cyan]Code Snippets:[/bold cyan]")
            for i, result in enumerate(results[:3], 1):  # Show top 3
                metadata = result.metadata if hasattr(result, 'metadata') else result
                code = metadata.get('code', '')
                name = metadata.get('name', 'Unknown')
                
                if code:
                    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                    self.console.print(Panel(
                        syntax,
                        title=f"{i}. {name}",
                        border_style="blue"
                    ))


@click.command()
@click.argument('query')
@click.option('-k', '--top-k', default=10, help='Number of results to show')
@click.option('-t', '--type', 'result_type', type=click.Choice(['function', 'class']),
              help='Filter by function or class')
@click.option('--code', is_flag=True, help='Show code snippets')
@click.option('--vector-weight', default=0.6, type=float, 
              help='Weight for vector search (0-1)')
@click.option('--keyword-weight', default=0.4, type=float,
              help='Weight for keyword search (0-1)')
@click.option('--no-latency', is_flag=True, help='Hide latency info')
@click.option('--json', 'json_output', is_flag=True, help='Output as JSON')
def search_command(query: str, 
                   top_k: int,
                   result_type: Optional[str],
                   code: bool,
                   vector_weight: float,
                   keyword_weight: float,
                   no_latency: bool,
                   json_output: bool):
    """Search for code using semantic and keyword search.
    
    Examples:
        codemind search "functions that fetch data"
        codemind search "greet user" --code --top-k 5
        codemind search "calculate" --type function --vector-weight 0.7
    """
    
    # Validate weights
    if not (0 <= vector_weight <= 1):
        click.echo("Error: vector-weight must be between 0 and 1", err=True)
        return
    if not (0 <= keyword_weight <= 1):
        click.echo("Error: keyword-weight must be between 0 and 1", err=True)
        return
    if abs((vector_weight + keyword_weight) - 1.0) > 0.01:
        click.echo("Warning: weights sum to {:.2f}, not 1.0".format(
            vector_weight + keyword_weight), err=True)
    
    try:
        engine = QueryEngine()
        
        # Ensure code is indexed
        if not engine.ensure_indexed():
            return
        
        # Execute search
        results, from_cache, search_time = engine.search(query, top_k=top_k, 
                                                         vector_weight=vector_weight,
                                                         keyword_weight=keyword_weight)
        
        # Filter by type if requested
        if result_type:
            results = engine.filter_results(results, result_type)
        
        # Record search in history
        engine.query_history.add(
            query=query,
            results_count=len(results),
            latency_ms=search_time * 1000,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            top_k=top_k,
            result_type=result_type,
            from_cache=from_cache
        )
        
        # Output results
        if json_output:
            # Convert to JSON-serializable format
            output = {
                'query': query,
                'from_cache': from_cache,
                'results_count': len(results),
                'latency_ms': search_time * 1000,
                'cache_stats': engine.query_cache.get_stats(),
                'results': [
                    {
                        'rank': i + 1,
                        'name': r.metadata.get('name') if hasattr(r, 'metadata') else 'Unknown',
                        'type': r.metadata.get('type') if hasattr(r, 'metadata') else 'unknown',
                        'file': r.metadata.get('file') if hasattr(r, 'metadata') else 'Unknown',
                        'score': r.combined_score if hasattr(r, 'combined_score') else 0,
                        'docstring': r.metadata.get('docstring') if hasattr(r, 'metadata') else ''
                    }
                    for i, r in enumerate(results)
                ]
            }
            click.echo(json.dumps(output, indent=2))
        else:
            # Display formatted results
            engine.display_results(results, query, show_code=code, 
                                  show_latency=not no_latency)
            
            # Show latency and cache status if not disabled
            if not no_latency:
                cache_indicator = "[cache]" if from_cache else ""
                click.echo(f"\nSearch completed in {search_time*1000:.2f}ms {cache_indicator}")
                
                # Show cache stats
                stats = engine.query_cache.get_stats()
                click.echo(f"Cache: {stats['cache_size']}/{stats['max_size']} | "
                         f"Hit rate: {stats['hit_rate_percent']}%")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == '__main__':
    search_command()
