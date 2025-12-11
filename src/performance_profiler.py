"""
Performance profiler for CodeMind query execution.

Profiles all major components to identify bottlenecks:
- Vector search latency
- Keyword search latency
- Hybrid search latency
- Cache performance
- Result filtering/sorting
- TUI rendering
"""

import time
import cProfile
import pstats
import io
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.cli.query import QueryEngine


@dataclass
class ProfileResult:
    """Individual profile measurement."""
    operation: str
    duration_ms: float
    memory_mb: float = 0.0
    call_count: int = 1


@dataclass
class ProfileReport:
    """Complete performance profile report."""
    results: List[ProfileResult] = field(default_factory=list)
    bottlenecks: List[Tuple[str, float]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def add_result(self, operation: str, duration_ms: float, memory_mb: float = 0.0):
        """Add a profile result."""
        self.results.append(ProfileResult(operation, duration_ms, memory_mb))

    def identify_bottlenecks(self, threshold_pct: float = 10.0):
        """Identify operations consuming >threshold_pct of time."""
        if not self.results:
            return

        total_time = sum(r.duration_ms for r in self.results)
        for result in self.results:
            pct = (result.duration_ms / total_time) * 100 if total_time > 0 else 0
            if pct >= threshold_pct:
                self.bottlenecks.append((result.operation, result.duration_ms))

        # Sort by duration
        self.bottlenecks.sort(key=lambda x: x[1], reverse=True)

    def print_report(self):
        """Print formatted performance report."""
        print("\n" + "="*60)
        print("CODEMIND PERFORMANCE PROFILE REPORT")
        print("="*60)

        if self.results:
            print("\nDetailed Timings:")
            print("-" * 60)
            print(f"{'Operation':<40} {'Time (ms)':>10} {'Calls':>8}")
            print("-" * 60)

            for result in sorted(self.results, key=lambda r: r.duration_ms, reverse=True):
                print(f"{result.operation:<40} {result.duration_ms:>10.2f} {result.call_count:>8}")

            total = sum(r.duration_ms for r in self.results)
            print("-" * 60)
            print(f"{'TOTAL':<40} {total:>10.2f}")

        if self.bottlenecks:
            print("\nIdentified Bottlenecks (>10% of total time):")
            print("-" * 60)
            total_bottleneck = sum(b[1] for b in self.bottlenecks)
            for op, duration in self.bottlenecks:
                pct = (duration / total_bottleneck) * 100
                print(f"  • {op}: {duration:.2f}ms ({pct:.1f}%)")

        if self.recommendations:
            print("\nOptimization Recommendations:")
            print("-" * 60)
            for i, rec in enumerate(self.recommendations, 1):
                print(f"  {i}. {rec}")

        print("\n" + "="*60 + "\n")


class PerformanceProfiler:
    """Profile CodeMind performance across all components."""

    def __init__(self, code_dir: str = "tests/fixtures"):
        """Initialize profiler.
        
        Args:
            code_dir: Directory containing code to search
        """
        self.code_dir = code_dir
        self.engine = QueryEngine()
        # Ensure code is indexed before profiling
        if not self.engine.ensure_indexed():
            raise RuntimeError("Failed to index code for profiling")
        self.report = ProfileReport()

    def profile_vector_search(self, query: str = "find file operations", iterations: int = 3) -> float:
        """Profile vector search latency via hybrid search with high vector weight."""
        print(f"Profiling vector-weighted search: '{query}' ({iterations} iterations)...", end=" ", flush=True)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            results, from_cache, latency = self.engine.search(
                query, 
                top_k=10,
                vector_weight=0.95,  # Primarily vector search
                keyword_weight=0.05
            )
            duration = (time.perf_counter() - start) * 1000  # Convert to ms

            times.append(duration)

        avg_time = sum(times) / len(times)
        self.report.add_result(f"Vector-Weighted Search ('{query}')", avg_time)
        print(f"avg {avg_time:.2f}ms")
        return avg_time

    def profile_keyword_search(self, query: str = "open file", iterations: int = 3) -> float:
        """Profile keyword search latency via hybrid search with high keyword weight."""
        print(f"Profiling keyword-weighted search: '{query}' ({iterations} iterations)...", end=" ", flush=True)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            results, from_cache, latency = self.engine.search(
                query,
                top_k=10,
                vector_weight=0.05,  # Primarily keyword search
                keyword_weight=0.95
            )
            duration = (time.perf_counter() - start) * 1000

            times.append(duration)

        avg_time = sum(times) / len(times)
        self.report.add_result(f"Keyword-Weighted Search ('{query}')", avg_time)
        print(f"avg {avg_time:.2f}ms")
        return avg_time

    def profile_hybrid_search(self, query: str = "handle errors", iterations: int = 3) -> float:
        """Profile hybrid search latency with balanced weights."""
        print(f"Profiling hybrid search: '{query}' ({iterations} iterations)...", end=" ", flush=True)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            results, from_cache, latency = self.engine.search(
                query,
                top_k=10,
                vector_weight=0.6,  # Balanced hybrid
                keyword_weight=0.4
            )
            duration = (time.perf_counter() - start) * 1000

            times.append(duration)

        avg_time = sum(times) / len(times)
        self.report.add_result(f"Hybrid Search ('{query}')", avg_time)
        print(f"avg {avg_time:.2f}ms")
        return avg_time

    def profile_cache_hit(self, query: str = "test cache", iterations: int = 5) -> float:
        """Profile cache hit performance."""
        print(f"Profiling cache hit: '{query}' ({iterations} iterations)...", end=" ", flush=True)

        # Warm up cache with first search
        self.engine.search(query)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            results, from_cache, latency = self.engine.search(query)
            duration = (time.perf_counter() - start) * 1000

            times.append(duration)

        avg_time = sum(times) / len(times)
        self.report.add_result(f"Cache Hit ('{query}')", avg_time)
        print(f"avg {avg_time:.2f}ms")
        return avg_time

    def profile_filtering(self, query: str = "filter", top_k: int = 50) -> float:
        """Profile result filtering latency."""
        print(f"Profiling filtering on {top_k} results...", end=" ", flush=True)

        # Get results to filter
        results = self.engine.search(query, top_k=top_k)[0]

        start = time.perf_counter()
        filtered = self.engine.advanced_filter(
            results,
            result_type="function",
            min_score=0.3,
            max_score=1.0
        )
        duration = (time.perf_counter() - start) * 1000

        self.report.add_result(f"Result Filtering ({top_k} results)", duration)
        print(f"{duration:.2f}ms")
        return duration

    def profile_sorting(self, query: str = "sort", top_k: int = 50) -> float:
        """Profile result sorting latency."""
        print(f"Profiling sorting on {top_k} results...", end=" ", flush=True)

        # Get results to sort
        results = self.engine.search(query, top_k=top_k)[0]

        start = time.perf_counter()
        sorted_results = self.engine.sort_results(results, sort_by="relevance")
        duration = (time.perf_counter() - start) * 1000

        self.report.add_result(f"Result Sorting ({top_k} results)", duration)
        print(f"{duration:.2f}ms")
        return duration

    def profile_with_cprofile(self, query: str = "test", iterations: int = 5):
        """Profile using Python's cProfile for detailed analysis."""
        print(f"\nDetailed profiling with cProfile ({iterations} iterations)...")

        profiler = cProfile.Profile()
        profiler.enable()

        for _ in range(iterations):
            self.engine.search(query)

        profiler.disable()

        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(15)  # Top 15 functions

        profile_output = s.getvalue()
        print("\nTop 15 Functions by Cumulative Time:")
        print("-" * 60)
        print(profile_output)

        return profile_output

    def run_full_profile(self) -> ProfileReport:
        """Run complete performance profile."""
        print("\n" + "="*60)
        print("STARTING CODEMIND PERFORMANCE PROFILE")
        print("="*60 + "\n")

        # Profile different search types
        self.profile_vector_search("find file operations")
        self.profile_vector_search("handle errors")

        self.profile_keyword_search("open file")
        self.profile_keyword_search("read write")

        self.profile_hybrid_search("handle errors")
        self.profile_hybrid_search("process data")

        # Profile caching
        self.profile_cache_hit("cached query")

        # Profile filtering and sorting
        self.profile_filtering()
        self.profile_sorting()

        # Analyze bottlenecks
        self.report.identify_bottlenecks(threshold_pct=5.0)

        # Generate recommendations
        self._generate_recommendations()

        # Print report
        self.report.print_report()

        return self.report

    def _generate_recommendations(self):
        """Generate optimization recommendations based on findings."""
        if not self.report.bottlenecks:
            return

        for op, duration in self.report.bottlenecks:
            if "Vector Search" in op and duration > 20:
                self.report.recommendations.append(
                    "Vector search is slow (>20ms). Consider: "
                    "1) Reduce embedding dimension, 2) Increase search_kwargs parameters, "
                    "3) Use faster embedding model"
                )
            elif "Keyword Search" in op and duration > 10:
                self.report.recommendations.append(
                    "Keyword search is slow (>10ms). Consider: "
                    "1) Optimize BM25 parameters, 2) Reduce corpus size, "
                    "3) Add more aggressive caching"
                )
            elif "Hybrid Search" in op and duration > 30:
                self.report.recommendations.append(
                    "Hybrid search is slow (>30ms). Prioritize optimizing component with highest latency "
                    "(vector or keyword). Consider parallel execution."
                )
            elif "Cache Hit" in op and duration > 1:
                self.report.recommendations.append(
                    "Cache hits are slower than expected (>1ms). Consider using in-memory only cache "
                    "for most frequent queries."
                )


if __name__ == "__main__":
    import sys

    code_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"
    profiler = PerformanceProfiler(code_dir)

    # Run full profile
    report = profiler.run_full_profile()

    # Run detailed cProfile analysis
    profiler.profile_with_cprofile()

    print("\nProfile complete!")
