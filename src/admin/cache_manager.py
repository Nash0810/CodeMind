"""
Cache Management Admin Interface

Provides tools for managing, monitoring, and optimizing caches.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.search.optimizations import EmbeddingCache, VectorSearchOptimizer
from src.query_cache import QueryCache
from src.query_history import QueryHistory


@dataclass
class CacheStats:
    """Statistics for a cache."""
    name: str
    type: str  # "query", "embedding", "file"
    size: int
    max_size: int
    hits: int
    misses: int
    hit_rate_percent: float
    memory_mb: float


class CacheManager:
    """Manage all caches in the system."""
    
    def __init__(self,
                 query_cache: QueryCache,
                 optimizer: VectorSearchOptimizer,
                 history: QueryHistory,
                 cache_dir: str = ".codemind_cache"):
        """Initialize cache manager.
        
        Args:
            query_cache: QueryCache instance
            optimizer: VectorSearchOptimizer instance
            history: QueryHistory instance
            cache_dir: Cache directory
        """
        self.query_cache = query_cache
        self.optimizer = optimizer
        self.history = history
        self.cache_dir = Path(cache_dir)
        self.console = Console()
    
    def get_all_stats(self) -> List[CacheStats]:
        """Get statistics for all caches.
        
        Returns:
            List of CacheStats
        """
        stats = []
        
        # Query cache stats
        if self.query_cache:
            stats.append(CacheStats(
                name="Query Results",
                type="query",
                size=self._query_cache_size(),
                max_size=100,
                hits=self.query_cache.hits,
                misses=self.query_cache.misses,
                hit_rate_percent=self._get_hit_rate(self.query_cache),
                memory_mb=self._query_cache_memory_mb()
            ))
        
        # Embedding cache stats
        if self.optimizer:
            emb_stats = self.optimizer.get_embedding_stats()
            stats.append(CacheStats(
                name="Embeddings",
                type="embedding",
                size=emb_stats['size'],
                max_size=emb_stats['max_size'],
                hits=emb_stats['hits'],
                misses=emb_stats['misses'],
                hit_rate_percent=emb_stats['hit_rate_percent'],
                memory_mb=self._embedding_cache_memory_mb(emb_stats['size'])
            ))
        
        # History stats
        if self.history:
            history_entries = len(self.history.get_recent(10000))
            stats.append(CacheStats(
                name="Search History",
                type="history",
                size=history_entries,
                max_size=1000,
                hits=0,
                misses=0,
                hit_rate_percent=100.0,
                memory_mb=self._history_memory_mb(history_entries)
            ))
        
        return stats
    
    def display_stats(self):
        """Display cache statistics in formatted table."""
        stats = self.get_all_stats()
        
        table = Table(title="Cache Statistics")
        table.add_column("Cache", style="cyan")
        table.add_column("Size", style="magenta")
        table.add_column("Hit Rate", style="green")
        table.add_column("Memory (MB)", style="yellow")
        table.add_column("Status", style="blue")
        
        for cache_stat in stats:
            status = self._get_status(cache_stat)
            table.add_row(
                cache_stat.name,
                f"{cache_stat.size}/{cache_stat.max_size}",
                f"{cache_stat.hit_rate_percent:.1f}%",
                f"{cache_stat.memory_mb:.2f}",
                status
            )
        
        self.console.print(table)
        
        # Summary
        total_memory = sum(s.memory_mb for s in stats)
        total_hits = sum(s.hits for s in stats)
        total_requests = sum(s.hits + s.misses for s in stats)
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        self.console.print(
            Panel(
                f"Total Memory: {total_memory:.2f} MB\n"
                f"Overall Hit Rate: {overall_hit_rate:.1f}%\n"
                f"Total Requests: {total_requests:,}",
                title="[bold]Summary[/bold]"
            )
        )
    
    def clear_cache(self, cache_type: str = "all"):
        """Clear specified cache(s).
        
        Args:
            cache_type: "query", "embedding", "history", or "all"
        """
        if cache_type in ("query", "all") and self.query_cache:
            self.query_cache.clear()
            self.console.print("[green]✓ Query cache cleared[/green]")
        
        if cache_type in ("embedding", "all") and self.optimizer:
            self.optimizer.embedding_cache.clear()
            self.console.print("[green]✓ Embedding cache cleared[/green]")
        
        if cache_type in ("history", "all") and self.history:
            self.history.clear()
            self.console.print("[green]✓ History cleared[/green]")
    
    def optimize_caches(self) -> Dict:
        """Analyze and optimize cache configuration.
        
        Returns:
            Dictionary with recommendations
        """
        stats = self.get_all_stats()
        recommendations = {
            'query_cache': [],
            'embedding_cache': [],
            'overall': []
        }
        
        for cache_stat in stats:
            if cache_stat.type == "query":
                # Query cache recommendations
                if cache_stat.hit_rate_percent < 30:
                    recommendations['query_cache'].append(
                        "Low hit rate - consider increasing cache size or TTL"
                    )
                if cache_stat.size >= cache_stat.max_size * 0.9:
                    recommendations['query_cache'].append(
                        "Cache near capacity - increase max_size or decrease TTL"
                    )
            
            elif cache_stat.type == "embedding":
                # Embedding cache recommendations
                if cache_stat.hit_rate_percent < 40:
                    recommendations['embedding_cache'].append(
                        "Low hit rate - consider increasing cache size"
                    )
                if cache_stat.hit_rate_percent > 80:
                    recommendations['embedding_cache'].append(
                        "Very high hit rate - cache is well-tuned"
                    )
        
        # Overall recommendations
        total_memory = sum(s.memory_mb for s in stats)
        if total_memory > 100:
            recommendations['overall'].append(
                "High memory usage - consider smaller caches or lazy loading"
            )
        
        return recommendations
    
    def display_recommendations(self):
        """Display optimization recommendations."""
        recommendations = self.optimize_caches()
        
        console = Console()
        
        for cache_type, recs in recommendations.items():
            if recs:
                console.print(f"\n[bold]{cache_type.replace('_', ' ').title()}:[/bold]")
                for i, rec in enumerate(recs, 1):
                    console.print(f"  {i}. {rec}")
        
        if not any(recommendations.values()):
            console.print("[green]✓ All caches are well-optimized![/green]")
    
    def export_stats(self, filepath: str):
        """Export cache statistics to JSON.
        
        Args:
            filepath: Output file path
        """
        stats = self.get_all_stats()
        data = {
            'timestamp': datetime.now().isoformat(),
            'caches': [asdict(s) for s in stats],
            'total_memory_mb': sum(s.memory_mb for s in stats),
            'overall_hit_rate': self._calculate_overall_hit_rate(stats)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.console.print(f"[green]✓ Statistics exported to {filepath}[/green]")
    
    # Helper methods
    
    @staticmethod
    def _get_hit_rate(cache) -> float:
        """Calculate cache hit rate."""
        total = cache.hits + cache.misses
        if total == 0:
            return 0.0
        return (cache.hits / total) * 100
    
    @staticmethod
    def _get_status(cache_stat: CacheStats) -> str:
        """Get status indicator for cache."""
        if cache_stat.hit_rate_percent >= 70:
            return "[green]✓ Optimal[/green]"
        elif cache_stat.hit_rate_percent >= 40:
            return "[yellow]⚠ Fair[/yellow]"
        else:
            return "[red]✗ Poor[/red]"
    
    @staticmethod
    def _calculate_overall_hit_rate(stats: List[CacheStats]) -> float:
        """Calculate overall hit rate across all caches."""
        total_hits = sum(s.hits for s in stats)
        total_requests = sum(s.hits + s.misses for s in stats)
        if total_requests == 0:
            return 0.0
        return (total_hits / total_requests) * 100
    
    @staticmethod
    def _query_cache_size() -> int:
        """Get query cache size (entries)."""
        # In real implementation, would query actual cache
        return 0
    
    @staticmethod
    def _query_cache_memory_mb() -> float:
        """Estimate query cache memory usage."""
        # Each cached result: ~5KB on average
        # 100 entries * 5KB = 500KB
        return 0.5
    
    @staticmethod
    def _embedding_cache_memory_mb(size: int) -> float:
        """Calculate embedding cache memory usage."""
        # Each embedding: 384-dim float32 = ~1.5KB
        # Plus overhead
        return (size * 1.5) / 1024 + 0.1
    
    @staticmethod
    def _history_memory_mb(entries: int) -> float:
        """Estimate history memory usage."""
        # Each history entry: ~200 bytes
        return (entries * 200) / (1024 * 1024)


class CacheMonitor:
    """Monitor cache performance over time."""
    
    def __init__(self, cache_manager: CacheManager, history_file: str = ".cache_monitor_history"):
        """Initialize monitor.
        
        Args:
            cache_manager: CacheManager instance
            history_file: File to store monitoring history
        """
        self.cache_manager = cache_manager
        self.history_file = history_file
        self.history = self._load_history()
    
    def record_snapshot(self):
        """Record current cache state."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'stats': [asdict(s) for s in self.cache_manager.get_all_stats()]
        }
        self.history.append(snapshot)
        self._save_history()
    
    def get_trend(self, metric: str = 'hit_rate_percent', window: int = 10) -> List[float]:
        """Get trend for a metric.
        
        Args:
            metric: Metric name ('hit_rate_percent', 'memory_mb', etc.)
            window: Number of recent snapshots
            
        Returns:
            List of metric values
        """
        trend = []
        for snapshot in self.history[-window:]:
            total = sum(s[metric] for s in snapshot['stats'])
            avg = total / len(snapshot['stats']) if snapshot['stats'] else 0
            trend.append(avg)
        return trend
    
    def _load_history(self) -> List[Dict]:
        """Load monitoring history from file."""
        if Path(self.history_file).exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save monitoring history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


# ==============================================================================
# CLI Commands
# ==============================================================================

import click


@click.group()
def cache_cli():
    """Cache management commands."""
    pass


@cache_cli.command()
def stats():
    """Show cache statistics."""
    from src.cli.production_engine import ProductionQueryEngine
    
    engine = ProductionQueryEngine()
    manager = CacheManager(
        engine.query_cache,
        engine.vector_search_optimizer,
        engine.query_history
    )
    manager.display_stats()


@cache_cli.command()
@click.option('-t', '--type', default='all', type=click.Choice(['query', 'embedding', 'history', 'all']))
def clear(type):
    """Clear caches."""
    from src.cli.production_engine import ProductionQueryEngine
    
    engine = ProductionQueryEngine()
    manager = CacheManager(
        engine.query_cache,
        engine.vector_search_optimizer,
        engine.query_history
    )
    manager.clear_cache(type)


@cache_cli.command()
def optimize():
    """Show optimization recommendations."""
    from src.cli.production_engine import ProductionQueryEngine
    
    engine = ProductionQueryEngine()
    manager = CacheManager(
        engine.query_cache,
        engine.vector_search_optimizer,
        engine.query_history
    )
    manager.display_recommendations()


@cache_cli.command()
@click.option('-o', '--output', required=True, help='Output file')
def export(output):
    """Export cache statistics."""
    from src.cli.production_engine import ProductionQueryEngine
    
    engine = ProductionQueryEngine()
    manager = CacheManager(
        engine.query_cache,
        engine.vector_search_optimizer,
        engine.query_history
    )
    manager.export_stats(output)


if __name__ == '__main__':
    cache_cli()
