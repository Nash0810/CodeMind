"""
Production QueryEngine with integrated optimizations.

Combines all Week 3 components (caching, filtering, sorting, optimizations)
into a single high-performance search interface.
"""

import click
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
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
from src.search.optimizations import (
    VectorSearchOptimizer,
    ResultBatcher,
    QueryNormalizer,
    LazyCodeLoader,
    EmbeddingCache
)


@dataclass
class SearchConfig:
    """Configuration for optimized search."""
    use_query_cache: bool = True
    use_embedding_cache: bool = True
    use_query_expansion: bool = False
    use_result_batching: bool = False
    use_lazy_loading: bool = True
    embedding_cache_size: int = 500
    query_cache_ttl: int = 3600
    batch_size: int = 10


class ProductionQueryEngine:
    """
    Production-ready query engine with all optimizations integrated.
    
    Features:
    - Multi-level caching (query + embedding)
    - Advanced result filtering (6 types)
    - Flexible sorting (5 criteria)
    - Persistent history with export
    - Query normalization and expansion
    - Result batching for progressive rendering
    - Lazy code loading for memory efficiency
    - Comprehensive statistics and monitoring
    """
    
    def __init__(self, 
                 persist_dir: str = "./chroma_db",
                 cache_dir: str = ".codemind_cache",
                 code_dir: str = ".",
                 config: Optional[SearchConfig] = None):
        """Initialize production query engine.
        
        Args:
            persist_dir: ChromaDB persistence directory
            cache_dir: Cache and history directory
            code_dir: Root code directory for lazy loading
            config: SearchConfig for optimization settings
        """
        self.persist_dir = persist_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.code_dir = code_dir
        self.config = config or SearchConfig()
        self.console = Console()
        
        # Core search components
        self.vector_store = VectorStore(persist_directory=persist_dir)
        self.keyword_search = KeywordSearch()
        self.hybrid_search = HybridSearch(self.vector_store, self.keyword_search)
        
        # Caching components
        if self.config.use_query_cache:
            self.query_cache = QueryCache(
                max_size=100,
                cache_dir=str(self.cache_dir / "queries")
            )
        else:
            self.query_cache = None
        
        # Optimization components
        if self.config.use_embedding_cache:
            self.vector_search_optimizer = VectorSearchOptimizer(
                self.vector_store,
                embedding_cache_size=self.config.embedding_cache_size
            )
        else:
            self.vector_search_optimizer = None
        
        self.result_batcher = ResultBatcher()
        self.query_normalizer = QueryNormalizer()
        
        if self.config.use_lazy_loading:
            self.code_loader = LazyCodeLoader(code_dir)
        else:
            self.code_loader = None
        
        # History tracking
        self.query_history = QueryHistory(
            history_file=str(self.cache_dir / "history.json")
        )
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'embedding_cache_hits': 0,
            'total_latency': 0.0,
            'expansion_searches': 0
        }
        
        self._indexed = False
    
    def ensure_indexed(self, code_json: str = "code_structure.json") -> bool:
        """Ensure code is indexed before searching.
        
        Args:
            code_json: Path to code_structure.json
            
        Returns:
            True if indexing successful
        """
        if self._indexed:
            return True
        
        if not Path(code_json).exists():
            self.console.print(f"[red]Error:[/red] {code_json} not found")
            return False
        
        if self.vector_store.count() == 0:
            self.console.print("[cyan]Indexing code...[/cyan]")
            try:
                with open(code_json) as f:
                    code_data = json.load(f)
                
                self.vector_store.index_code_blocks(code_data)
                self.keyword_search.index_code_blocks(code_data)
                self.console.print(
                    f"[green]Indexed {self.vector_store.count()} code blocks[/green]"
                )
            except Exception as e:
                self.console.print(f"[red]Indexing failed: {e}[/red]")
                return False
        
        self._indexed = True
        return True
    
    def search(self,
               query: str,
               top_k: int = 10,
               vector_weight: float = 0.6,
               keyword_weight: float = 0.4,
               use_expansion: Optional[bool] = None,
               return_batches: bool = False) -> Tuple[List[Dict], Dict]:
        """Execute optimized search.
        
        Args:
            query: Search query
            top_k: Number of results
            vector_weight: Vector search weight
            keyword_weight: Keyword search weight
            use_expansion: Override config for expansion (None = use config)
            return_batches: Return results as batches
            
        Returns:
            (results, search_metadata)
            where search_metadata contains stats about the search
        """
        if not self._indexed:
            raise RuntimeError("Code not indexed. Call ensure_indexed() first")
        
        start_time = time.time()
        self.stats['total_searches'] += 1
        
        metadata = {
            'query': query,
            'top_k': top_k,
            'from_cache': False,
            'from_embedding_cache': False,
            'used_expansion': False,
            'latency_ms': 0.0,
            'result_count': 0,
            'cache_stats': {}
        }
        
        # Determine expansion strategy
        use_expansion = use_expansion if use_expansion is not None else self.config.use_query_expansion
        
        # Step 1: Check query result cache
        if self.config.use_query_cache and self.query_cache:
            cached_results = self.query_cache.get(query, vector_weight, keyword_weight, top_k)
            if cached_results:
                results, _ = cached_results
                self.stats['cache_hits'] += 1
                metadata['from_cache'] = True
                metadata['result_count'] = len(results)
                self._record_search(query, results, metadata)
                
                if return_batches:
                    return (self.result_batcher.batch_results(results, self.config.batch_size), metadata)
                return (results, metadata)
        
        # Step 2: Execute search (with or without expansion)
        if use_expansion and self.vector_search_optimizer:
            results = self.vector_search_optimizer.search_with_expansion(query, top_k)
            self.stats['expansion_searches'] += 1
            metadata['used_expansion'] = True
        elif self.config.use_embedding_cache and self.vector_search_optimizer:
            results, from_embedding_cache = self.vector_search_optimizer.search_with_cache(query, top_k)
            if from_embedding_cache:
                self.stats['embedding_cache_hits'] += 1
            metadata['from_embedding_cache'] = from_embedding_cache
        else:
            # Standard hybrid search
            results = self.hybrid_search.search(
                query,
                top_k=top_k,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight
            )
        
        # Step 3: Cache results
        if self.config.use_query_cache and self.query_cache:
            self.query_cache.set(query, results, vector_weight, keyword_weight, top_k)
        
        # Step 4: Record search and get final stats
        latency = time.time() - start_time
        metadata['latency_ms'] = latency * 1000
        metadata['result_count'] = len(results)
        
        if self.vector_search_optimizer:
            metadata['cache_stats'] = self.vector_search_optimizer.get_embedding_stats()
        
        self.stats['total_latency'] += latency
        self._record_search(query, results, metadata)
        
        if return_batches:
            return (self.result_batcher.batch_results(results, self.config.batch_size), metadata)
        return (results, metadata)
    
    def advanced_search(self,
                       query: str,
                       filters: Optional[Dict] = None,
                       sort_by: Optional[str] = None,
                       group_by: Optional[str] = None,
                       top_k: int = 10) -> Tuple[List[Dict], Dict]:
        """Advanced search with filtering, sorting, and grouping.
        
        Args:
            query: Search query
            filters: Dict of filter parameters:
                {
                    'type': 'function'/'class',
                    'file': 'pattern',
                    'docstring': 'term',
                    'min_score': 0.3,
                    'max_score': 1.0,
                    'code_pattern': 'regex'
                }
            sort_by: Sort order ('relevance', 'name', 'file', 'type', 'latency')
            group_by: Group by field ('type', 'file', 'name')
            top_k: Number of results
            
        Returns:
            (results, metadata)
        """
        # Get base results
        results, metadata = self.search(query, top_k=top_k*2)  # Get more to filter
        
        # Apply filters
        if filters:
            filter_obj = ResultFilter()
            
            if 'type' in filters:
                filter_obj.by_type(filters['type'])
            if 'file' in filters:
                filter_obj.by_file(filters['file'])
            if 'docstring' in filters:
                filter_obj.by_docstring(filters['docstring'])
            if 'min_score' in filters or 'max_score' in filters:
                min_s = filters.get('min_score', 0.0)
                max_s = filters.get('max_score', 1.0)
                filter_obj.by_score_range(min_s, max_s)
            if 'code_pattern' in filters:
                filter_obj.by_code_pattern(filters['code_pattern'])
            
            results = filter_obj.apply(results)
        
        # Apply sorting
        if sort_by:
            sorter = ResultSorter()
            sorter.sort(results, SortOrder[sort_by.upper()])
        
        # Trim to top_k
        results = results[:top_k]
        
        # Apply grouping (returns dict)
        if group_by:
            return (self.result_batcher.group_by_field(results, group_by), metadata)
        
        metadata['result_count'] = len(results)
        return (results, metadata)
    
    def get_code(self, filepath: str, start_line: int = 0, end_line: Optional[int] = None) -> str:
        """Get code content with lazy loading.
        
        Args:
            filepath: Path to file
            start_line: Starting line
            end_line: Ending line
            
        Returns:
            Code content
        """
        if self.code_loader:
            return self.code_loader.get_code(filepath, start_line, end_line)
        return ""
    
    def get_statistics(self) -> Dict:
        """Get comprehensive search statistics.
        
        Returns:
            Dictionary with stats
        """
        total = self.stats['total_searches']
        if total == 0:
            return {'message': 'No searches performed'}
        
        return {
            'total_searches': total,
            'cache_hit_rate': (self.stats['cache_hits'] / total) * 100,
            'embedding_cache_hit_rate': (self.stats['embedding_cache_hits'] / total) * 100,
            'expansion_searches': self.stats['expansion_searches'],
            'avg_latency_ms': (self.stats['total_latency'] / total) * 1000,
            'history_entries': len(self.query_history.get_recent(1000)),
        }
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent search history.
        
        Args:
            limit: Number of recent searches
            
        Returns:
            List of history entries
        """
        return self.query_history.get_recent(limit)
    
    def _record_search(self, query: str, results: List[Dict], metadata: Dict):
        """Record search in history.
        
        Args:
            query: Search query
            results: Results returned
            metadata: Search metadata
        """
        self.query_history.add(
            query=query,
            result_count=len(results),
            latency_ms=metadata['latency_ms'],
            metadata={
                'from_cache': metadata['from_cache'],
                'from_embedding_cache': metadata['from_embedding_cache'],
                'used_expansion': metadata['used_expansion']
            }
        )
    
    def clear_cache(self):
        """Clear all caches."""
        if self.query_cache:
            self.query_cache.clear()
        if self.vector_search_optimizer:
            self.vector_search_optimizer.embedding_cache.clear()
        if self.code_loader:
            self.code_loader.clear_cache()
    
    def export_history(self, filepath: str, format: str = 'json'):
        """Export search history.
        
        Args:
            filepath: Output file path
            format: 'json' or 'csv'
        """
        if format == 'json':
            self.query_history.export_json(filepath)
        elif format == 'csv':
            self.query_history.export_csv(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")


# ==============================================================================
# CLI Interface
# ==============================================================================

@click.group()
def cli():
    """CodeMind CLI - Production search interface."""
    pass


@cli.command()
@click.argument('query')
@click.option('-k', '--top-k', default=10, help='Number of results')
@click.option('-v', '--vector-weight', default=0.6, type=float, help='Vector search weight')
@click.option('-w', '--keyword-weight', default=0.4, type=float, help='Keyword search weight')
@click.option('--expand', is_flag=True, help='Use query expansion')
@click.option('--no-cache', is_flag=True, help='Disable caching')
@click.option('--json', is_flag=True, help='JSON output')
def search(query, top_k, vector_weight, keyword_weight, expand, no_cache, json_output):
    """Search code with optimizations."""
    
    # Create engine
    config = SearchConfig(
        use_query_cache=not no_cache,
        use_embedding_cache=not no_cache,
        use_query_expansion=expand
    )
    engine = ProductionQueryEngine(config=config)
    
    # Index and search
    if not engine.ensure_indexed():
        return
    
    results, metadata = engine.search(
        query,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        use_expansion=expand
    )
    
    # Display results
    if json_output:
        click.echo(json.dumps(results, indent=2))
    else:
        console = Console()
        
        # Header
        status = "[green]CACHED[/green]" if metadata['from_cache'] else "[yellow]LIVE[/yellow]"
        console.print(
            f"\n[bold]Results for:[/bold] {query} {status}\n"
        )
        
        # Results table
        table = Table(title=f"Found {len(results)} results")
        table.add_column("Name", style="cyan")
        table.add_column("File", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("Score", style="yellow")
        
        for result in results[:top_k]:
            meta = result.get('metadata', {})
            table.add_row(
                meta.get('name', 'Unknown'),
                meta.get('file', 'Unknown'),
                meta.get('type', 'Unknown'),
                f"{result.get('score', 0):.3f}"
            )
        
        console.print(table)
        
        # Metadata
        console.print(
            f"\n[dim]Latency: {metadata['latency_ms']:.2f}ms | "
            f"Cache: {metadata['from_cache']} | "
            f"Expansion: {metadata['used_expansion']}[/dim]"
        )


@cli.command()
def stats():
    """Show search statistics."""
    engine = ProductionQueryEngine()
    stats = engine.get_statistics()
    
    console = Console()
    console.print(Panel.fit(
        json.dumps(stats, indent=2),
        title="[bold]Search Statistics[/bold]"
    ))


@cli.command()
def history():
    """Show search history."""
    engine = ProductionQueryEngine()
    
    if not engine.ensure_indexed():
        return
    
    recent = engine.get_history(10)
    
    console = Console()
    table = Table(title="Recent Searches")
    table.add_column("Query", style="cyan")
    table.add_column("Results", style="magenta")
    table.add_column("Latency (ms)", style="yellow")
    table.add_column("Cached", style="green")
    
    for entry in recent:
        table.add_row(
            entry.get('query', 'Unknown'),
            str(entry.get('result_count', 0)),
            f"{entry.get('latency_ms', 0):.2f}",
            "Yes" if entry.get('metadata', {}).get('from_cache') else "No"
        )
    
    console.print(table)


@cli.command()
@click.option('-f', '--format', default='json', type=click.Choice(['json', 'csv']))
@click.option('-o', '--output', required=True, help='Output file')
def export_history(format, output):
    """Export search history."""
    engine = ProductionQueryEngine()
    engine.export_history(output, format=format)
    
    console = Console()
    console.print(f"[green]History exported to {output}[/green]")


if __name__ == '__main__':
    cli()
