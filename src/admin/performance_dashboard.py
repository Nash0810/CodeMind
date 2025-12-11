"""
Performance Monitoring Dashboard

Real-time monitoring of search performance, latency, cache effectiveness.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
import statistics

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, DownloadColumn
from rich.align import Align
from rich.text import Text


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: datetime
    operation: str  # "search", "filter", "sort", "cache_hit", etc.
    latency_ms: float
    cache_hit: bool
    result_count: int


@dataclass
class PerformanceReport:
    """Aggregated performance report."""
    total_operations: int
    total_latency_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    cache_hit_rate: float
    operations_per_second: float
    total_results_returned: int


class PerformanceMonitor:
    """Monitor and analyze search performance."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize performance monitor.
        
        Args:
            window_size: Number of recent metrics to keep
        """
        self.metrics = deque(maxlen=window_size)
        self.window_size = window_size
        self.start_time = datetime.now()
        self.console = Console()
    
    def record_metric(self,
                     operation: str,
                     latency_ms: float,
                     cache_hit: bool = False,
                     result_count: int = 0):
        """Record a performance metric.
        
        Args:
            operation: Type of operation
            latency_ms: Operation latency in milliseconds
            cache_hit: Whether this was a cache hit
            result_count: Number of results returned
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            operation=operation,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            result_count=result_count
        )
        self.metrics.append(metric)
    
    def get_report(self) -> PerformanceReport:
        """Generate performance report.
        
        Returns:
            PerformanceReport with aggregated metrics
        """
        if not self.metrics:
            return PerformanceReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        latencies = [m.latency_ms for m in self.metrics]
        cache_hits = sum(1 for m in self.metrics if m.cache_hit)
        
        uptime = datetime.now() - self.start_time
        uptime_seconds = uptime.total_seconds()
        ops_per_second = len(self.metrics) / uptime_seconds if uptime_seconds > 0 else 0
        
        return PerformanceReport(
            total_operations=len(self.metrics),
            total_latency_ms=sum(latencies),
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=self._percentile(latencies, 50),
            p95_latency_ms=self._percentile(latencies, 95),
            p99_latency_ms=self._percentile(latencies, 99),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            cache_hit_rate=(cache_hits / len(self.metrics) * 100) if self.metrics else 0,
            operations_per_second=ops_per_second,
            total_results_returned=sum(m.result_count for m in self.metrics)
        )
    
    def display_dashboard(self):
        """Display performance dashboard."""
        report = self.get_report()
        
        # Main metrics
        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Total Operations", f"{report.total_operations:,}")
        metrics_table.add_row("Average Latency", f"{report.avg_latency_ms:.2f} ms")
        metrics_table.add_row("P50 Latency", f"{report.p50_latency_ms:.2f} ms")
        metrics_table.add_row("P95 Latency", f"{report.p95_latency_ms:.2f} ms")
        metrics_table.add_row("P99 Latency", f"{report.p99_latency_ms:.2f} ms")
        metrics_table.add_row("Min Latency", f"{report.min_latency_ms:.2f} ms")
        metrics_table.add_row("Max Latency", f"{report.max_latency_ms:.2f} ms")
        metrics_table.add_row("Cache Hit Rate", f"{report.cache_hit_rate:.1f}%")
        metrics_table.add_row("Operations/Second", f"{report.operations_per_second:.2f}")
        
        self.console.print(metrics_table)
        
        # Operation breakdown
        self._display_operation_breakdown()
        
        # Latency distribution
        self._display_latency_distribution()
        
        # Summary panel
        summary_text = f"""
[bold blue]Uptime:[/bold blue] {self._format_uptime()}
[bold blue]Total Results:[/bold blue] {report.total_results_returned:,}
[bold blue]Total Time Spent:[/bold blue] {report.total_latency_ms/1000:.2f}s
"""
        self.console.print(Panel(summary_text, title="[bold]Summary[/bold]"))
    
    def get_operation_stats(self) -> Dict[str, Dict]:
        """Get statistics by operation type.
        
        Returns:
            Dictionary mapping operation names to statistics
        """
        ops = {}
        
        for metric in self.metrics:
            if metric.operation not in ops:
                ops[metric.operation] = {
                    'count': 0,
                    'latencies': [],
                    'cache_hits': 0,
                    'total_results': 0
                }
            
            ops[metric.operation]['count'] += 1
            ops[metric.operation]['latencies'].append(metric.latency_ms)
            if metric.cache_hit:
                ops[metric.operation]['cache_hits'] += 1
            ops[metric.operation]['total_results'] += metric.result_count
        
        # Calculate aggregates
        for op_name in ops:
            latencies = ops[op_name]['latencies']
            count = ops[op_name]['count']
            
            ops[op_name]['avg_latency'] = statistics.mean(latencies)
            ops[op_name]['p95_latency'] = self._percentile(latencies, 95)
            ops[op_name]['cache_hit_rate'] = (ops[op_name]['cache_hits'] / count * 100) if count else 0
        
        return ops
    
    def display_operation_stats(self):
        """Display statistics by operation type."""
        ops = self.get_operation_stats()
        
        table = Table(title="Operation Statistics")
        table.add_column("Operation", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Avg (ms)", style="yellow")
        table.add_column("P95 (ms)", style="yellow")
        table.add_column("Cache Hit %", style="green")
        
        for op_name in sorted(ops.keys()):
            op_stats = ops[op_name]
            table.add_row(
                op_name,
                str(op_stats['count']),
                f"{op_stats['avg_latency']:.2f}",
                f"{op_stats['p95_latency']:.2f}",
                f"{op_stats['cache_hit_rate']:.1f}%"
            )
        
        self.console.print(table)
    
    def get_percentile(self, percentile: int) -> float:
        """Get latency percentile.
        
        Args:
            percentile: Percentile (0-100)
            
        Returns:
            Latency at percentile
        """
        latencies = [m.latency_ms for m in self.metrics]
        return self._percentile(latencies, percentile) if latencies else 0
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON.
        
        Args:
            filepath: Output file path
        """
        report = self.get_report()
        ops = self.get_operation_stats()
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'report': asdict(report),
            'operation_stats': {
                op: {k: v for k, v in stats.items() if k != 'latencies'}
                for op, stats in ops.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.console.print(f"[green]✓ Metrics exported to {filepath}[/green]")
    
    # Helper methods
    
    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        lower = int(index)
        upper = lower + 1
        
        if upper >= len(sorted_values):
            return sorted_values[-1]
        
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    
    def _format_uptime(self) -> str:
        """Format uptime as human-readable string."""
        uptime = datetime.now() - self.start_time
        total_seconds = int(uptime.total_seconds())
        
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0:
            parts.append(f"{seconds}s")
        
        return " ".join(parts) if parts else "0s"
    
    def _display_operation_breakdown(self):
        """Display operations breakdown as table."""
        ops = self.get_operation_stats()
        
        table = Table(title="Operation Breakdown")
        table.add_column("Operation", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Avg Latency", style="yellow")
        table.add_column("Cache Hit Rate", style="green")
        
        for op_name in sorted(ops.keys()):
            op_stats = ops[op_name]
            table.add_row(
                op_name,
                f"{op_stats['count']}",
                f"{op_stats['avg_latency']:.2f} ms",
                f"{op_stats['cache_hit_rate']:.1f}%"
            )
        
        self.console.print(table)
    
    def _display_latency_distribution(self):
        """Display latency distribution visualization."""
        latencies = [m.latency_ms for m in self.metrics]
        if not latencies:
            return
        
        # Create histogram
        min_lat = min(latencies)
        max_lat = max(latencies)
        
        # Create 10 buckets
        bucket_size = (max_lat - min_lat) / 10 if max_lat > min_lat else 1
        buckets = [0] * 10
        
        for lat in latencies:
            bucket_idx = min(int((lat - min_lat) / bucket_size), 9)
            buckets[bucket_idx] += 1
        
        max_count = max(buckets)
        
        # Display histogram
        self.console.print("\n[bold]Latency Distribution:[/bold]")
        for i, count in enumerate(buckets):
            bucket_min = min_lat + i * bucket_size
            bucket_max = bucket_min + bucket_size
            bar_width = int(40 * count / max_count) if max_count > 0 else 0
            bar = "█" * bar_width
            self.console.print(f"{bucket_min:6.1f}-{bucket_max:6.1f}ms │ {bar} {count}")


class PerformanceThresholdAnalyzer:
    """Analyze performance against thresholds."""
    
    def __init__(self, monitor: PerformanceMonitor):
        """Initialize analyzer.
        
        Args:
            monitor: PerformanceMonitor instance
        """
        self.monitor = monitor
        self.thresholds = {
            'avg_latency_ms': 50.0,
            'p95_latency_ms': 100.0,
            'p99_latency_ms': 500.0,
            'cache_hit_rate': 50.0,
        }
        self.console = Console()
    
    def set_threshold(self, metric: str, value: float):
        """Set performance threshold.
        
        Args:
            metric: Metric name
            value: Threshold value
        """
        if metric in self.thresholds:
            self.thresholds[metric] = value
    
    def check_thresholds(self) -> Dict[str, bool]:
        """Check if performance meets thresholds.
        
        Returns:
            Dictionary mapping metric names to pass/fail
        """
        report = self.monitor.get_report()
        
        results = {
            'avg_latency_ok': report.avg_latency_ms <= self.thresholds['avg_latency_ms'],
            'p95_latency_ok': report.p95_latency_ms <= self.thresholds['p95_latency_ms'],
            'p99_latency_ok': report.p99_latency_ms <= self.thresholds['p99_latency_ms'],
            'cache_hit_ok': report.cache_hit_rate >= self.thresholds['cache_hit_rate'],
        }
        
        return results
    
    def display_report(self):
        """Display threshold compliance report."""
        results = self.check_thresholds()
        report = self.monitor.get_report()
        
        table = Table(title="Threshold Compliance")
        table.add_column("Metric", style="cyan")
        table.add_column("Threshold", style="magenta")
        table.add_column("Actual", style="yellow")
        table.add_column("Status", style="green")
        
        # Average latency
        status = "✓ PASS" if results['avg_latency_ok'] else "✗ FAIL"
        table.add_row(
            "Avg Latency",
            f"{self.thresholds['avg_latency_ms']:.1f} ms",
            f"{report.avg_latency_ms:.2f} ms",
            status
        )
        
        # P95 latency
        status = "✓ PASS" if results['p95_latency_ok'] else "✗ FAIL"
        table.add_row(
            "P95 Latency",
            f"{self.thresholds['p95_latency_ms']:.1f} ms",
            f"{report.p95_latency_ms:.2f} ms",
            status
        )
        
        # P99 latency
        status = "✓ PASS" if results['p99_latency_ok'] else "✗ FAIL"
        table.add_row(
            "P99 Latency",
            f"{self.thresholds['p99_latency_ms']:.1f} ms",
            f"{report.p99_latency_ms:.2f} ms",
            status
        )
        
        # Cache hit rate
        status = "✓ PASS" if results['cache_hit_ok'] else "✗ FAIL"
        table.add_row(
            "Cache Hit Rate",
            f"{self.thresholds['cache_hit_rate']:.1f}%",
            f"{report.cache_hit_rate:.1f}%",
            status
        )
        
        self.console.print(table)
        
        # Overall status
        all_pass = all(results.values())
        status_text = "[green]✓ ALL THRESHOLDS MET[/green]" if all_pass else "[red]✗ THRESHOLD VIOLATIONS[/red]"
        self.console.print(Panel(status_text, title="[bold]Overall Status[/bold]"))


# ==============================================================================
# CLI Commands
# ==============================================================================

import click


@click.group()
def monitoring_cli():
    """Performance monitoring commands."""
    pass


@monitoring_cli.command()
def dashboard():
    """Display performance dashboard."""
    from src.cli.production_engine import ProductionQueryEngine
    
    engine = ProductionQueryEngine()
    # In real implementation, would pass actual monitor from engine
    monitor = PerformanceMonitor()
    monitor.display_dashboard()


@monitoring_cli.command()
def operations():
    """Show operation-level statistics."""
    from src.cli.production_engine import ProductionQueryEngine
    
    engine = ProductionQueryEngine()
    monitor = PerformanceMonitor()
    monitor.display_operation_stats()


@monitoring_cli.command()
@click.option('-o', '--output', required=True, help='Output file')
def export(output):
    """Export performance metrics."""
    from src.cli.production_engine import ProductionQueryEngine
    
    engine = ProductionQueryEngine()
    monitor = PerformanceMonitor()
    monitor.export_metrics(output)


if __name__ == '__main__':
    monitoring_cli()
