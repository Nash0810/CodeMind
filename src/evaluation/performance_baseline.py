"""Performance baseline measurement for CodeMind parser."""

import time
import json
from pathlib import Path
from typing import Dict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser.directory_walker import walk_directory
from parser.call_graph import CallGraph


def measure_parsing_performance(repo_path: str) -> Dict:
    """
    Measures baseline parsing performance on a repository.
    
    Returns detailed metrics for optimization targeting.
    
    Args:
        repo_path: Path to repository to measure
        
    Returns:
        Dictionary with performance metrics
    """
    print("="*60)
    print("PERFORMANCE BASELINE MEASUREMENT")
    print("="*60)
    print(f"Repository: {repo_path}\n")
    
    # Count files first
    repo = Path(repo_path)
    py_files = list(repo.rglob('*.py'))
    total_lines = 0
    for f in py_files:
        try:
            with open(f) as file:
                total_lines += len(file.readlines())
        except:
            pass
    
    print(f"Python files: {len(py_files)}")
    print(f"Total lines of code: {total_lines:,}\n")
    
    # Measure parsing time
    start = time.time()
    results = walk_directory(repo_path)
    parse_time = time.time() - start
    
    # Measure call graph construction
    start = time.time()
    graph = CallGraph()
    graph.build_from_files(results)
    graph_time = time.time() - start
    
    # Calculate metrics
    total_functions = sum(len(f.functions) for f in results)
    total_classes = sum(len(f.classes) for f in results)
    
    metrics = {
        'repository': repo_path,
        'files_parsed': len(results),
        'total_lines': total_lines,
        'functions_extracted': total_functions,
        'classes_extracted': total_classes,
        'parse_time_seconds': round(parse_time, 2),
        'graph_time_seconds': round(graph_time, 2),
        'total_time_seconds': round(parse_time + graph_time, 2),
        'lines_per_second': round(total_lines / parse_time) if parse_time > 0 else 0,
        'functions_per_second': round(total_functions / parse_time) if parse_time > 0 else 0,
        'avg_time_per_file_ms': round((parse_time / len(results)) * 1000, 2) if results else 0
    }
    
    # Print results
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    for key, value in metrics.items():
        if key == 'repository':
            continue
        print(f"{key}: {value}")
    
    # Performance assessment
    print("\n" + "="*60)
    print("PERFORMANCE ASSESSMENT")
    print("="*60)
    
    if metrics['lines_per_second'] < 10000:
        print("⚠️  SLOW: Parsing <10K lines/sec")
        print("   Target for optimization: Parser hot loop")
        print("   Suggested approach: Cython or Rust rewrite")
    elif metrics['lines_per_second'] < 50000:
        print("⚙️  MODERATE: Parsing 10K-50K lines/sec")
        print("   Room for improvement with optimization")
    else:
        print("✅ FAST: Parsing >50K lines/sec")
    
    if metrics['avg_time_per_file_ms'] > 100:
        print(f"\n⚠️  File parsing is slow (>{metrics['avg_time_per_file_ms']}ms per file)")
        print("   Consider parallelization with multiprocessing")
    
    # Save metrics
    with open('performance_baseline.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Baseline saved to: performance_baseline.json")
    
    return metrics


def compare_performance(before_path: str, after_path: str):
    """
    Compare two performance baseline measurements.
    
    Args:
        before_path: Path to baseline JSON before optimization
        after_path: Path to baseline JSON after optimization
    """
    with open(before_path) as f:
        before = json.load(f)
    
    with open(after_path) as f:
        after = json.load(f)
    
    print("="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    metrics_to_compare = [
        'parse_time_seconds',
        'lines_per_second',
        'functions_per_second',
        'avg_time_per_file_ms'
    ]
    
    for metric in metrics_to_compare:
        before_val = before[metric]
        after_val = after[metric]
        
        # Calculate improvement
        if 'time' in metric or 'ms' in metric:
            # Lower is better
            improvement = ((before_val - after_val) / before_val) * 100
            speedup = before_val / after_val if after_val > 0 else 0
        else:
            # Higher is better
            improvement = ((after_val - before_val) / before_val) * 100
            speedup = after_val / before_val if before_val > 0 else 0
        
        print(f"\n{metric}:")
        print(f"  Before: {before_val}")
        print(f"  After: {after_val}")
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"  Speedup: {speedup:.2f}x")
