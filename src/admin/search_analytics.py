"""
Search Analytics System

Track and analyze search patterns, popular queries, and optimization opportunities.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.bar import Bar


@dataclass
class QueryAnalysis:
    """Analysis of a single query."""
    query: str
    frequency: int
    avg_latency_ms: float
    cache_hit_rate: float
    avg_results_returned: int
    success_rate: float
    last_used: datetime


class SearchAnalytics:
    """Analyze search patterns and usage."""
    
    def __init__(self, history_file: str = ".codemind_cache/analytics.json"):
        """Initialize analytics system.
        
        Args:
            history_file: File to persist analytics data
        """
        self.history_file = Path(history_file)
        self.data = self._load_data()
        self.console = Console()
    
    def record_search(self,
                     query: str,
                     latency_ms: float,
                     cache_hit: bool,
                     results_count: int,
                     success: bool = True):
        """Record a search operation.
        
        Args:
            query: Search query
            latency_ms: Operation latency
            cache_hit: Whether it was a cache hit
            results_count: Number of results returned
            success: Whether search succeeded
        """
        if 'queries' not in self.data:
            self.data['queries'] = {}
        
        query_key = query.lower()
        if query_key not in self.data['queries']:
            self.data['queries'][query_key] = {
                'query': query,
                'executions': [],
                'cache_hits': 0,
                'total_results': 0,
                'successful': 0,
                'failed': 0
            }
        
        query_data = self.data['queries'][query_key]
        query_data['executions'].append({
            'timestamp': datetime.now().isoformat(),
            'latency_ms': latency_ms,
            'cache_hit': cache_hit,
            'results_count': results_count
        })
        
        # Keep only last 100 executions per query
        query_data['executions'] = query_data['executions'][-100:]
        
        if cache_hit:
            query_data['cache_hits'] += 1
        
        query_data['total_results'] += results_count
        
        if success:
            query_data['successful'] += 1
        else:
            query_data['failed'] += 1
        
        self._save_data()
    
    def get_popular_queries(self, limit: int = 20) -> List[QueryAnalysis]:
        """Get most frequently executed queries.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of QueryAnalysis sorted by frequency
        """
        analyses = []
        
        if 'queries' not in self.data:
            return analyses
        
        for query_key, query_data in self.data['queries'].items():
            executions = query_data['executions']
            if not executions:
                continue
            
            frequency = len(executions)
            latencies = [e['latency_ms'] for e in executions]
            cache_hits = query_data['cache_hits']
            
            analysis = QueryAnalysis(
                query=query_data['query'],
                frequency=frequency,
                avg_latency_ms=statistics.mean(latencies),
                cache_hit_rate=(cache_hits / frequency * 100) if frequency else 0,
                avg_results_returned=query_data['total_results'] // frequency if frequency else 0,
                success_rate=100 * query_data['successful'] / (query_data['successful'] + query_data['failed']) if (query_data['successful'] + query_data['failed']) > 0 else 0,
                last_used=datetime.fromisoformat(executions[-1]['timestamp']) if executions else datetime.now()
            )
            analyses.append(analysis)
        
        # Sort by frequency
        analyses.sort(key=lambda x: x.frequency, reverse=True)
        return analyses[:limit]
    
    def get_slowest_queries(self, limit: int = 20) -> List[QueryAnalysis]:
        """Get slowest queries.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of QueryAnalysis sorted by latency
        """
        analyses = self.get_popular_queries(limit=10000)  # Get all
        analyses.sort(key=lambda x: x.avg_latency_ms, reverse=True)
        return analyses[:limit]
    
    def get_low_cache_hit_queries(self, threshold: float = 30.0, limit: int = 20) -> List[QueryAnalysis]:
        """Get queries with low cache hit rates.
        
        Args:
            threshold: Cache hit rate threshold (%)
            limit: Maximum number of queries to return
            
        Returns:
            List of QueryAnalysis with low cache hit rates
        """
        analyses = self.get_popular_queries(limit=10000)  # Get all
        low_hit = [a for a in analyses if a.cache_hit_rate < threshold]
        low_hit.sort(key=lambda x: x.cache_hit_rate)
        return low_hit[:limit]
    
    def display_popular_queries(self, limit: int = 10):
        """Display popular queries table."""
        queries = self.get_popular_queries(limit)
        
        table = Table(title="Popular Queries")
        table.add_column("Query", style="cyan")
        table.add_column("Frequency", style="magenta")
        table.add_column("Avg Latency", style="yellow")
        table.add_column("Cache Hit %", style="green")
        table.add_column("Avg Results", style="blue")
        
        for analysis in queries:
            table.add_row(
                analysis.query[:50],
                str(analysis.frequency),
                f"{analysis.avg_latency_ms:.2f} ms",
                f"{analysis.cache_hit_rate:.1f}%",
                str(analysis.avg_results_returned)
            )
        
        self.console.print(table)
    
    def display_slowest_queries(self, limit: int = 10):
        """Display slowest queries table."""
        queries = self.get_slowest_queries(limit)
        
        table = Table(title="Slowest Queries")
        table.add_column("Query", style="cyan")
        table.add_column("Avg Latency", style="red")
        table.add_column("Frequency", style="magenta")
        table.add_column("Cache Hit %", style="green")
        
        for analysis in queries:
            table.add_row(
                analysis.query[:50],
                f"{analysis.avg_latency_ms:.2f} ms",
                str(analysis.frequency),
                f"{analysis.cache_hit_rate:.1f}%"
            )
        
        self.console.print(table)
    
    def display_recommendations(self):
        """Display optimization recommendations based on analytics."""
        console = Console()
        
        # Check for slow queries
        slow = self.get_slowest_queries(limit=100)
        if slow and slow[0].avg_latency_ms > 50:
            console.print("\n[bold red]Slow Queries:[/bold red]")
            for q in slow[:5]:
                console.print(f"  • {q.query[:60]} - {q.avg_latency_ms:.1f}ms")
                if q.cache_hit_rate < 30:
                    console.print(f"    → Consider caching or query optimization")
                else:
                    console.print(f"    → Cache hit rate good ({q.cache_hit_rate:.0f}%), may need query restructuring")
        
        # Check for low cache hit queries
        low_cache = self.get_low_cache_hit_queries(threshold=30)
        if low_cache:
            console.print("\n[bold yellow]Low Cache Hit Rate:[/bold yellow]")
            for q in low_cache[:5]:
                console.print(f"  • {q.query[:60]} - {q.cache_hit_rate:.0f}% hit rate")
                console.print(f"    → Query varies slightly; consider normalization")
        
        # Check for high-frequency queries
        popular = self.get_popular_queries(limit=100)
        frequent = [q for q in popular if q.frequency > 50]
        if frequent:
            console.print("\n[bold cyan]High Frequency Queries:[/bold cyan]")
            for q in frequent[:5]:
                console.print(f"  • {q.query[:60]} - {q.frequency} uses")
                if q.cache_hit_rate < 50:
                    console.print(f"    → Pre-cache results for better performance")
        
        console.print("\n[bold green]✓ Analysis complete[/bold green]")
    
    def get_statistics(self) -> Dict:
        """Get overall search statistics.
        
        Returns:
            Dictionary with various statistics
        """
        if 'queries' not in self.data or not self.data['queries']:
            return {
                'total_queries': 0,
                'total_executions': 0,
                'avg_latency_ms': 0,
                'overall_cache_hit_rate': 0,
                'most_common_intent': None
            }
        
        total_executions = 0
        total_latency = 0
        total_cache_hits = 0
        intents = Counter()
        
        for query_key, query_data in self.data['queries'].items():
            executions = query_data['executions']
            total_executions += len(executions)
            total_latency += sum(e['latency_ms'] for e in executions)
            total_cache_hits += query_data['cache_hits']
            
            # Infer intent from query
            intent = self._infer_intent(query_data['query'])
            intents[intent] += len(executions)
        
        return {
            'total_queries': len(self.data['queries']),
            'total_executions': total_executions,
            'avg_latency_ms': total_latency / total_executions if total_executions else 0,
            'overall_cache_hit_rate': (total_cache_hits / total_executions * 100) if total_executions else 0,
            'most_common_intent': intents.most_common(1)[0][0] if intents else None,
            'queries_per_hour': self._get_queries_per_hour(),
            'peak_hour': self._get_peak_hour()
        }
    
    def display_statistics(self):
        """Display overall statistics."""
        stats = self.get_statistics()
        
        stats_table = Table(title="Search Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Unique Queries", str(stats['total_queries']))
        stats_table.add_row("Total Executions", f"{stats['total_executions']:,}")
        stats_table.add_row("Average Latency", f"{stats['avg_latency_ms']:.2f} ms")
        stats_table.add_row("Overall Cache Hit Rate", f"{stats['overall_cache_hit_rate']:.1f}%")
        stats_table.add_row("Most Common Intent", stats['most_common_intent'] or "Unknown")
        stats_table.add_row("Queries/Hour", f"{stats['queries_per_hour']:.1f}")
        
        self.console.print(stats_table)
    
    def get_trends(self, hours: int = 24) -> Dict:
        """Get search trends over time.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with trend data
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        hourly_counts = defaultdict(int)
        hourly_latency = defaultdict(list)
        
        if 'queries' not in self.data:
            return {'hourly_counts': dict(hourly_counts), 'hourly_avg_latency': {}}
        
        for query_data in self.data['queries'].values():
            for execution in query_data['executions']:
                exec_time = datetime.fromisoformat(execution['timestamp'])
                if exec_time > cutoff_time:
                    hour_key = exec_time.strftime('%Y-%m-%d %H:00')
                    hourly_counts[hour_key] += 1
                    hourly_latency[hour_key].append(execution['latency_ms'])
        
        # Calculate averages
        hourly_avg_latency = {
            k: statistics.mean(v) for k, v in hourly_latency.items()
        }
        
        return {
            'hourly_counts': dict(hourly_counts),
            'hourly_avg_latency': hourly_avg_latency
        }
    
    def export_analytics(self, filepath: str):
        """Export analytics to JSON.
        
        Args:
            filepath: Output file path
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'popular_queries': [asdict(q) for q in self.get_popular_queries(limit=100)],
            'slowest_queries': [asdict(q) for q in self.get_slowest_queries(limit=50)],
            'trends': self.get_trends(hours=24)
        }
        
        # Convert datetime objects to strings
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=serialize_datetime)
        
        self.console.print(f"[green]✓ Analytics exported to {filepath}[/green]")
    
    # Helper methods
    
    @staticmethod
    def _infer_intent(query: str) -> str:
        """Infer search intent from query.
        
        Args:
            query: Search query
            
        Returns:
            Intent category
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['def ', 'function', 'method']):
            return 'find_function'
        elif any(word in query_lower for word in ['class ', 'type']):
            return 'find_class'
        elif any(word in query_lower for word in ['use', 'call', 'import', 'from']):
            return 'find_usage'
        elif any(word in query_lower for word in ['pattern', 'regex', 'match']):
            return 'find_pattern'
        elif any(word in query_lower for word in ['how', 'what', 'why', 'explain']):
            return 'explain'
        else:
            return 'generic'
    
    def _get_queries_per_hour(self) -> float:
        """Calculate average queries per hour."""
        if 'queries' not in self.data:
            return 0
        
        # Get time range
        all_times = []
        for query_data in self.data['queries'].values():
            for execution in query_data['executions']:
                all_times.append(datetime.fromisoformat(execution['timestamp']))
        
        if len(all_times) < 2:
            return 0
        
        earliest = min(all_times)
        latest = max(all_times)
        hours = (latest - earliest).total_seconds() / 3600
        
        return len(all_times) / hours if hours > 0 else 0
    
    def _get_peak_hour(self) -> str:
        """Get peak hour of day.
        
        Returns:
            Hour in HH:00 format with most queries
        """
        if 'queries' not in self.data:
            return "Unknown"
        
        hourly_counts = defaultdict(int)
        
        for query_data in self.data['queries'].values():
            for execution in query_data['executions']:
                exec_time = datetime.fromisoformat(execution['timestamp'])
                hour_key = exec_time.strftime('%H:00')
                hourly_counts[hour_key] += 1
        
        if not hourly_counts:
            return "Unknown"
        
        peak = max(hourly_counts.items(), key=lambda x: x[1])
        return peak[0]
    
    def _load_data(self) -> Dict:
        """Load analytics data from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_data(self):
        """Save analytics data to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)


# ==============================================================================
# CLI Commands
# ==============================================================================

import click


@click.group()
def analytics_cli():
    """Search analytics commands."""
    pass


@analytics_cli.command()
@click.option('-l', '--limit', default=10, help='Number of queries to show')
def popular(limit):
    """Show popular queries."""
    analytics = SearchAnalytics()
    analytics.display_popular_queries(limit)


@analytics_cli.command()
@click.option('-l', '--limit', default=10, help='Number of queries to show')
def slowest(limit):
    """Show slowest queries."""
    analytics = SearchAnalytics()
    analytics.display_slowest_queries(limit)


@analytics_cli.command()
def stats():
    """Show search statistics."""
    analytics = SearchAnalytics()
    analytics.display_statistics()


@analytics_cli.command()
def recommend():
    """Show optimization recommendations."""
    analytics = SearchAnalytics()
    analytics.display_recommendations()


@analytics_cli.command()
@click.option('-o', '--output', required=True, help='Output file')
def export(output):
    """Export analytics data."""
    analytics = SearchAnalytics()
    analytics.export_analytics(output)


if __name__ == '__main__':
    analytics_cli()
