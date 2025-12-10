"""Performance profiling infrastructure for CodeMind parser."""

import cProfile
import pstats
import io
from pathlib import Path
from typing import Callable, Any


class ParserProfiler:
    """Wrapper for profiling parser performance"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """
        Profiles a function call and returns its result.
        
        Args:
            func: Function to profile
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function call
        """
        self.profiler.enable()
        result = func(*args, **kwargs)
        self.profiler.disable()
        
        return result
    
    def print_stats(self, sort_by: str = 'cumtime', top_n: int = 20):
        """
        Prints profiling statistics to console.
        
        Args:
            sort_by: Sort key ('cumtime', 'tottime', 'ncalls')
            top_n: Number of top functions to display
        """
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(top_n)
        print(s.getvalue())
    
    def save_stats(self, output_path: str):
        """
        Saves profiling data to file for visualization with snakeviz.
        
        Args:
            output_path: Path to save .prof file
        """
        self.profiler.dump_stats(output_path)
        print(f"Profile saved to: {output_path}")
        print(f"Visualize with: snakeviz {output_path}")


def profile_parser(repo_path: str) -> dict:
    """
    Profiles parsing a repository and returns statistics.
    
    Args:
        repo_path: Path to repository to profile
        
    Returns:
        Dict with timing information and bottleneck analysis
    """
    from .directory_walker import walk_directory
    import time
    
    profiler = ParserProfiler()
    
    start = time.time()
    results = profiler.profile_function(walk_directory, repo_path)
    elapsed = time.time() - start
    
    # Print detailed stats
    print("\n" + "="*60)
    print("PROFILING RESULTS")
    print("="*60)
    profiler.print_stats(sort_by='cumtime', top_n=15)
    
    # Save for visualization
    profiler.save_stats('parser_profile.prof')
    
    # Calculate statistics
    total_functions = sum(len(f.functions) for f in results)
    total_classes = sum(len(f.classes) for f in results)
    
    stats = {
        'total_time': elapsed,
        'files_parsed': len(results),
        'functions_extracted': total_functions,
        'classes_extracted': total_classes,
        'avg_time_per_file': elapsed / len(results) if results else 0,
        'functions_per_second': total_functions / elapsed if elapsed > 0 else 0
    }
    
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    return stats
