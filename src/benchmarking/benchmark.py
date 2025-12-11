"""
Comprehensive benchmarking suite for CodeMind search vs grep.

This module compares the performance of CodeMind's hybrid search
against grep-based search across multiple metrics:
- Recall: Did we find the relevant code?
- Precision: How many false positives?
- F1 Score: Balanced metric
- Latency: How fast was the search?
"""

import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import os

from src.indexing.vector_store import VectorStore
from src.indexing.keyword_search import KeywordSearch
from src.search.hybrid_search import HybridSearch
from src.benchmarking.grep_baseline import GrepBaseline


@dataclass
class BenchmarkQuery:
    """A benchmark query with ground truth results."""
    query: str
    description: str
    expected_functions: List[str]  # Function names that should be found
    expected_classes: List[str]    # Class names that should be found


# Curated test queries with known expected results
BENCHMARK_QUERIES = [
    BenchmarkQuery(
        query="greet user",
        description="Find functions that greet or interact with users",
        expected_functions=["greet", "hello_world", "speak"],
        expected_classes=[]
    ),
    BenchmarkQuery(
        query="data transformation",
        description="Find functions that transform or process data",
        expected_functions=["transform", "get_info"],
        expected_classes=["DataProcessor"]
    ),
    BenchmarkQuery(
        query="fetch or retrieve data",
        description="Find functions that fetch or retrieve data",
        expected_functions=["fetch_data", "get_info"],
        expected_classes=[]
    ),
    BenchmarkQuery(
        query="calculate or compute",
        description="Find math or calculation functions",
        expected_functions=["calculate_sum", "add", "subtract"],
        expected_classes=["Calculator"]
    ),
    BenchmarkQuery(
        query="animal behavior",
        description="Find animal-related classes and methods",
        expected_functions=["speak"],
        expected_classes=["Animal", "Dog"]
    ),
]


@dataclass
class BenchmarkMetrics:
    """Metrics for a single search method."""
    method: str
    query: str
    latency_ms: float
    results_count: int
    recall: float
    precision: float
    f1_score: float
    found_items: List[str]


class CodeMindBenchmark:
    """Benchmarking suite comparing CodeMind vs grep."""

    def __init__(self, code_json_path: str = "code_structure.json"):
        """Initialize benchmark with CodeMind components.
        
        Args:
            code_json_path: Path to parsed code structure JSON
        """
        self.code_json_path = code_json_path
        
        # Load code structure
        with open(code_json_path, 'r') as f:
            self.code_data = json.load(f)
        
        # Initialize CodeMind components
        self.vector_store = VectorStore(persist_directory="./chroma_db")
        self.keyword_search = KeywordSearch()
        self.hybrid_search = HybridSearch(self.vector_store, self.keyword_search)
        
        # Initialize grep baseline
        self.grep = GrepBaseline()
        
        # Index the code
        self._ensure_indexed()

    def _ensure_indexed(self):
        """Ensure code is indexed in both systems."""
        # Check if already indexed by checking count
        if self.vector_store.count() == 0:
            print("[*] Indexing code with vector and keyword search...")
            blocks = self._extract_blocks()
            print(f"[*] Extracted {len(blocks)} blocks from code data")
            self.vector_store.index_code_blocks(self.code_data)
            self.keyword_search.index_code_blocks(self.code_data)
            print(f"[+] Indexed {len(blocks)} code blocks")
        else:
            print(f"[*] Already indexed {self.vector_store.count()} code blocks")

    def _extract_blocks(self) -> List[Dict]:
        """Extract code blocks from parsed code structure."""
        blocks = []
        for file_data in self.code_data:
            file_path = file_data.get('file') or file_data.get('file_path')
            
            # Functions
            for func in file_data.get('functions', []):
                blocks.append({
                    'file': file_path,
                    'name': func.get('name'),
                    'type': 'function',
                    'docstring': (func.get('docstring') or '')[:200],
                    'code': func.get('code', '')
                })
            
            # Classes
            for cls in file_data.get('classes', []):
                blocks.append({
                    'file': file_path,
                    'name': cls.get('name'),
                    'type': 'class',
                    'docstring': (cls.get('docstring') or '')[:200],
                    'code': cls.get('code', '')
                })
        
        return blocks

    def extract_item_names(self, results: List[Dict]) -> List[str]:
        """Extract function/class names from search results.
        
        Args:
            results: Search results with 'name' field in metadata
            
        Returns:
            List of item names
        """
        names = []
        for result in results:
            # Handle SearchResult dataclass from hybrid search
            if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                if 'name' in result.metadata:
                    names.append(result.metadata['name'])
            # Handle dict results
            elif isinstance(result, dict):
                if 'name' in result:
                    names.append(result['name'])
                elif 'metadata' in result and 'name' in result['metadata']:
                    names.append(result['metadata']['name'])
        
        return list(set(names))  # Deduplicate

    def calculate_recall(self, found: List[str], expected: List[str]) -> float:
        """Calculate recall: proportion of expected items found.
        
        Recall = True Positives / (True Positives + False Negatives)
        """
        if not expected:
            return 1.0
        
        true_positives = len([x for x in found if x in expected])
        return true_positives / len(expected)

    def calculate_precision(self, found: List[str], expected: List[str]) -> float:
        """Calculate precision: proportion of found items that are correct.
        
        Precision = True Positives / (True Positives + False Positives)
        """
        if not found:
            return 0.0
        
        true_positives = len([x for x in found if x in expected])
        return true_positives / len(found)

    def calculate_f1(self, recall: float, precision: float) -> float:
        """Calculate F1 score: harmonic mean of recall and precision."""
        if recall + precision == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def benchmark_codemind(self, query: BenchmarkQuery) -> BenchmarkMetrics:
        """Benchmark CodeMind hybrid search.
        
        Args:
            query: Benchmark query
            
        Returns:
            BenchmarkMetrics with performance data
        """
        start = time.time()
        results = self.hybrid_search.search(
            query.query,
            top_k=20,
            vector_weight=0.6,
            keyword_weight=0.4
        )
        latency = (time.time() - start) * 1000
        
        # Extract found items
        found_items = self.extract_item_names(
            [r.__dict__ if hasattr(r, '__dict__') else r for r in results]
        )
        
        # All expected items (functions + classes)
        all_expected = query.expected_functions + query.expected_classes
        
        # Calculate metrics
        recall = self.calculate_recall(found_items, all_expected)
        precision = self.calculate_precision(found_items, all_expected)
        f1 = self.calculate_f1(recall, precision)
        
        return BenchmarkMetrics(
            method='CodeMind (Hybrid)',
            query=query.query,
            latency_ms=latency,
            results_count=len(results),
            recall=recall,
            precision=precision,
            f1_score=f1,
            found_items=found_items
        )

    def benchmark_grep(self, query: BenchmarkQuery) -> BenchmarkMetrics:
        """Benchmark grep baseline.
        
        Uses function name search which is more semantically aware
        than plain keyword search.
        
        Args:
            query: Benchmark query
            
        Returns:
            BenchmarkMetrics with performance data
        """
        start = time.time()
        # Use function/class definition search
        results = self.grep.search_function_names(query.query, max_results=20)
        latency = (time.time() - start) * 1000
        
        # Extract function/class names from grep results
        found_items = []
        for result in results:
            # Try to extract function/class name
            import re
            match = re.search(r'(def|class)\s+(\w+)', result.line_content)
            if match:
                found_items.append(match.group(2))
        
        found_items = list(set(found_items))  # Deduplicate
        
        # All expected items
        all_expected = query.expected_functions + query.expected_classes
        
        # Calculate metrics
        recall = self.calculate_recall(found_items, all_expected)
        precision = self.calculate_precision(found_items, all_expected)
        f1 = self.calculate_f1(recall, precision)
        
        return BenchmarkMetrics(
            method='Grep Baseline',
            query=query.query,
            latency_ms=latency,
            results_count=len(results),
            recall=recall,
            precision=precision,
            f1_score=f1,
            found_items=found_items
        )

    def run_all_benchmarks(self) -> Dict:
        """Run all benchmark queries on all search methods.
        
        Returns:
            Dictionary with complete benchmark results
        """
        results = {
            'timestamp': time.time(),
            'queries': []
        }
        
        for query in BENCHMARK_QUERIES:
            print(f"\n[*] Benchmarking: {query.description}")
            print(f"    Query: '{query.query}'")
            
            # Test CodeMind
            codemind_metrics = self.benchmark_codemind(query)
            print(f"    CodeMind: {codemind_metrics.latency_ms:.2f}ms, "
                  f"recall={codemind_metrics.recall:.2f}, "
                  f"precision={codemind_metrics.precision:.2f}, "
                  f"f1={codemind_metrics.f1_score:.2f}")
            
            # Test Grep
            grep_metrics = self.benchmark_grep(query)
            print(f"    Grep:     {grep_metrics.latency_ms:.2f}ms, "
                  f"recall={grep_metrics.recall:.2f}, "
                  f"precision={grep_metrics.precision:.2f}, "
                  f"f1={grep_metrics.f1_score:.2f}")
            
            # Calculate improvement
            if grep_metrics.f1_score > 0:
                f1_improvement = ((codemind_metrics.f1_score - grep_metrics.f1_score) 
                                  / grep_metrics.f1_score * 100)
            else:
                f1_improvement = 0
            
            print(f"    F1 Improvement: {f1_improvement:+.1f}%")
            
            results['queries'].append({
                'query': query.query,
                'description': query.description,
                'expected_functions': query.expected_functions,
                'expected_classes': query.expected_classes,
                'codemind': asdict(codemind_metrics),
                'grep': asdict(grep_metrics),
                'f1_improvement_percent': f1_improvement
            })
        
        return results

    def print_summary(self, results: Dict):
        """Print benchmark summary.
        
        Args:
            results: Results from run_all_benchmarks()
        """
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY".center(70))
        print("="*70)
        
        # Aggregate metrics
        codemind_avg_f1 = 0
        codemind_avg_latency = 0
        grep_avg_f1 = 0
        grep_avg_latency = 0
        f1_improvements = []
        
        for query_result in results['queries']:
            codemind_metrics = query_result['codemind']
            grep_metrics = query_result['grep']
            
            codemind_avg_f1 += codemind_metrics['f1_score']
            codemind_avg_latency += codemind_metrics['latency_ms']
            grep_avg_f1 += grep_metrics['f1_score']
            grep_avg_latency += grep_metrics['latency_ms']
            f1_improvements.append(query_result['f1_improvement_percent'])
        
        n = len(results['queries'])
        codemind_avg_f1 /= n
        codemind_avg_latency /= n
        grep_avg_f1 /= n
        grep_avg_latency /= n
        avg_improvement = sum(f1_improvements) / len(f1_improvements)
        
        print(f"\nCodeMind Hybrid Search:")
        print(f"  Average F1 Score:  {codemind_avg_f1:.3f}")
        print(f"  Average Latency:   {codemind_avg_latency:.2f}ms")
        print(f"\nGrep Baseline:")
        print(f"  Average F1 Score:  {grep_avg_f1:.3f}")
        print(f"  Average Latency:   {grep_avg_latency:.2f}ms")
        print(f"\nImprovement:")
        print(f"  F1 Score Improvement: {avg_improvement:+.1f}%")
        print(f"  Latency Improvement:  {(grep_avg_latency - codemind_avg_latency)/codemind_avg_latency*100:.1f}%")
        print("="*70 + "\n")


if __name__ == "__main__":
    print("[*] Starting CodeMind Benchmarking Suite")
    
    try:
        benchmark = CodeMindBenchmark()
        results = benchmark.run_all_benchmarks()
        benchmark.print_summary(results)
        
        # Save results
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("[+] Results saved to benchmark_results.json")
        
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
