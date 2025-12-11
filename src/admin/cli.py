"""
Unified Admin CLI for CodeMind

Provides administrative tools for cache management, performance monitoring,
and search analytics.
"""

import click
from rich.console import Console
from rich.panel import Panel

from src.admin.cache_manager import CacheManager, CacheMonitor
from src.admin.performance_dashboard import PerformanceMonitor, PerformanceThresholdAnalyzer
from src.admin.search_analytics import SearchAnalytics


console = Console()


@click.group()
def admin_cli():
    """CodeMind Administrative Tools
    
    Manage caches, monitor performance, and analyze search patterns.
    """
    pass


# ==============================================================================
# Cache Management Commands
# ==============================================================================

@admin_cli.group()
def cache():
    """Cache management commands."""
    pass


@cache.command()
def stats():
    """Display cache statistics and health."""
    from src.cli.production_engine import ProductionQueryEngine
    
    try:
        engine = ProductionQueryEngine()
        manager = CacheManager(
            engine.query_cache,
            engine.vector_search_optimizer,
            engine.query_history
        )
        manager.display_stats()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cache.command()
@click.option('-t', '--type', default='all',
              type=click.Choice(['query', 'embedding', 'history', 'all']),
              help='Cache type to clear')
def clear(type):
    """Clear specified caches."""
    from src.cli.production_engine import ProductionQueryEngine
    
    try:
        engine = ProductionQueryEngine()
        manager = CacheManager(
            engine.query_cache,
            engine.vector_search_optimizer,
            engine.query_history
        )
        
        if click.confirm(f"Clear {type} cache(s)?"):
            manager.clear_cache(type)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cache.command()
def optimize():
    """Show cache optimization recommendations."""
    from src.cli.production_engine import ProductionQueryEngine
    
    try:
        engine = ProductionQueryEngine()
        manager = CacheManager(
            engine.query_cache,
            engine.vector_search_optimizer,
            engine.query_history
        )
        
        console.print(Panel("[bold]Cache Optimization Analysis[/bold]"))
        manager.display_recommendations()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cache.command()
@click.option('-o', '--output', required=True,
              help='Output file (JSON)')
def export(output):
    """Export cache statistics to file."""
    from src.cli.production_engine import ProductionQueryEngine
    
    try:
        engine = ProductionQueryEngine()
        manager = CacheManager(
            engine.query_cache,
            engine.vector_search_optimizer,
            engine.query_history
        )
        manager.export_stats(output)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ==============================================================================
# Performance Monitoring Commands
# ==============================================================================

@admin_cli.group()
def performance():
    """Performance monitoring commands."""
    pass


@performance.command()
def dashboard():
    """Display performance dashboard."""
    try:
        monitor = PerformanceMonitor()
        console.print(Panel("[bold]Performance Dashboard[/bold]"))
        monitor.display_dashboard()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@performance.command()
def operations():
    """Show operation-level performance statistics."""
    try:
        monitor = PerformanceMonitor()
        console.print(Panel("[bold]Operation Statistics[/bold]"))
        monitor.display_operation_stats()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@performance.command()
@click.option('-p', '--percentile', default=95, type=int,
              help='Latency percentile (0-100)')
def latency(percentile):
    """Show latency percentile."""
    try:
        monitor = PerformanceMonitor()
        value = monitor.get_percentile(percentile)
        console.print(f"P{percentile} Latency: {value:.2f} ms")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@performance.command()
def thresholds():
    """Check performance against thresholds."""
    try:
        monitor = PerformanceMonitor()
        analyzer = PerformanceThresholdAnalyzer(monitor)
        analyzer.display_report()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@performance.command()
@click.option('-o', '--output', required=True,
              help='Output file (JSON)')
def export_perf(output):
    """Export performance metrics to file."""
    try:
        monitor = PerformanceMonitor()
        monitor.export_metrics(output)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ==============================================================================
# Search Analytics Commands
# ==============================================================================

@admin_cli.group()
def analytics():
    """Search analytics commands."""
    pass


@analytics.command()
@click.option('-l', '--limit', default=10, help='Number of queries to show')
def popular(limit):
    """Show most popular queries."""
    try:
        analytics = SearchAnalytics()
        console.print(Panel("[bold]Popular Queries[/bold]"))
        analytics.display_popular_queries(limit)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@analytics.command()
@click.option('-l', '--limit', default=10, help='Number of queries to show')
def slowest(limit):
    """Show slowest performing queries."""
    try:
        analytics = SearchAnalytics()
        console.print(Panel("[bold]Slowest Queries[/bold]"))
        analytics.display_slowest_queries(limit)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@analytics.command()
def stats():
    """Show overall search statistics."""
    try:
        analytics = SearchAnalytics()
        console.print(Panel("[bold]Search Statistics[/bold]"))
        analytics.display_statistics()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@analytics.command()
def recommend():
    """Show optimization recommendations based on analytics."""
    try:
        analytics = SearchAnalytics()
        console.print(Panel("[bold]Analytics Recommendations[/bold]"))
        analytics.display_recommendations()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@analytics.command()
@click.option('-o', '--output', required=True,
              help='Output file (JSON)')
def export_analytics(output):
    """Export analytics data to file."""
    try:
        analytics = SearchAnalytics()
        analytics.export_analytics(output)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ==============================================================================
# System-wide Commands
# ==============================================================================

@admin_cli.command()
def health():
    """Show overall system health status."""
    from src.cli.production_engine import ProductionQueryEngine
    from rich.table import Table
    
    try:
        engine = ProductionQueryEngine()
        
        # Get all stats
        cache_mgr = CacheManager(
            engine.query_cache,
            engine.vector_search_optimizer,
            engine.query_history
        )
        perf_monitor = PerformanceMonitor()
        analytics = SearchAnalytics()
        
        # Create health report
        cache_stats = cache_mgr.get_all_stats()
        perf_report = perf_monitor.get_report()
        analytics_stats = analytics.get_statistics()
        
        # Display health table
        health_table = Table(title="System Health")
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="green")
        health_table.add_column("Details", style="yellow")
        
        # Cache health
        avg_hit_rate = sum(s.hit_rate_percent for s in cache_stats) / len(cache_stats) if cache_stats else 0
        cache_status = "✓ Healthy" if avg_hit_rate > 50 else "⚠ Fair" if avg_hit_rate > 30 else "✗ Poor"
        health_table.add_row("Caches", cache_status, f"Hit rate: {avg_hit_rate:.1f}%")
        
        # Performance health
        perf_status = "✓ Healthy" if perf_report.avg_latency_ms < 50 else "⚠ Fair" if perf_report.avg_latency_ms < 100 else "✗ Slow"
        health_table.add_row("Performance", perf_status, f"Avg latency: {perf_report.avg_latency_ms:.2f}ms")
        
        # Analytics health
        analytics_status = "✓ Active" if analytics_stats['total_executions'] > 0 else "⚠ Inactive"
        health_table.add_row("Analytics", analytics_status, f"Total queries: {analytics_stats['total_queries']}")
        
        console.print(health_table)
        
        # Overall status
        overall = "✓ Healthy" if all([avg_hit_rate > 50, perf_report.avg_latency_ms < 50]) else "⚠ Monitor"
        console.print(Panel(f"[bold]Overall Status: {overall}[/bold]"))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@admin_cli.command()
def report():
    """Generate comprehensive system report."""
    from datetime import datetime
    
    try:
        from src.cli.production_engine import ProductionQueryEngine
        
        engine = ProductionQueryEngine()
        
        # Collect all data
        cache_mgr = CacheManager(
            engine.query_cache,
            engine.vector_search_optimizer,
            engine.query_history
        )
        perf_monitor = PerformanceMonitor()
        analytics = SearchAnalytics()
        
        # Create report
        report_lines = [
            f"[bold]CodeMind System Report[/bold]",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "[bold cyan]Cache Statistics[/bold cyan]"
        ]
        
        for stat in cache_mgr.get_all_stats():
            report_lines.append(
                f"  {stat.name}: {stat.size}/{stat.max_size} "
                f"({stat.hit_rate_percent:.1f}% hit rate)"
            )
        
        perf_report = perf_monitor.get_report()
        report_lines.extend([
            "",
            "[bold cyan]Performance Metrics[/bold cyan]",
            f"  Total Operations: {perf_report.total_operations:,}",
            f"  Average Latency: {perf_report.avg_latency_ms:.2f}ms",
            f"  P95 Latency: {perf_report.p95_latency_ms:.2f}ms",
            f"  P99 Latency: {perf_report.p99_latency_ms:.2f}ms",
            f"  Cache Hit Rate: {perf_report.cache_hit_rate:.1f}%",
            f"  Operations/Second: {perf_report.operations_per_second:.2f}"
        ])
        
        analytics_stats = analytics.get_statistics()
        report_lines.extend([
            "",
            "[bold cyan]Analytics[/bold cyan]",
            f"  Unique Queries: {analytics_stats['total_queries']}",
            f"  Total Executions: {analytics_stats['total_executions']:,}",
            f"  Most Common Intent: {analytics_stats['most_common_intent']}",
            f"  Peak Hour: {analytics_stats.get('peak_hour', 'Unknown')}"
        ])
        
        # Display report
        for line in report_lines:
            console.print(line)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@admin_cli.command()
@click.option('-o', '--output', required=True,
              help='Output file (JSON)')
def full_export(output):
    """Export complete system data to file."""
    import json
    from pathlib import Path
    from datetime import datetime
    
    try:
        from src.cli.production_engine import ProductionQueryEngine
        
        engine = ProductionQueryEngine()
        
        cache_mgr = CacheManager(
            engine.query_cache,
            engine.vector_search_optimizer,
            engine.query_history
        )
        perf_monitor = PerformanceMonitor()
        analytics = SearchAnalytics()
        
        # Collect all data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'cache': {
                'stats': [
                    {
                        'name': s.name,
                        'size': s.size,
                        'max_size': s.max_size,
                        'hit_rate': s.hit_rate_percent
                    }
                    for s in cache_mgr.get_all_stats()
                ]
            },
            'performance': {
                'total_operations': perf_monitor.get_report().total_operations,
                'avg_latency_ms': perf_monitor.get_report().avg_latency_ms,
                'p95_latency_ms': perf_monitor.get_report().p95_latency_ms,
                'p99_latency_ms': perf_monitor.get_report().p99_latency_ms,
                'cache_hit_rate': perf_monitor.get_report().cache_hit_rate
            },
            'analytics': analytics.get_statistics()
        }
        
        with open(output, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        console.print(f"[green]✓ System data exported to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == '__main__':
    admin_cli()
