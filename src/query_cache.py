"""
Query result caching system for CodeMind.

Implements LRU cache for query results and embeddings to speed up repeated searches.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict

from src.search.hybrid_search import SearchResult


@dataclass
class CacheEntry:
    """A cached query result."""
    query: str
    results: List[Dict[str, Any]]
    timestamp: float
    vector_weight: float
    keyword_weight: float
    top_k: int
    result_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(**data)


class QueryCache:
    """LRU cache for query results."""
    
    def __init__(self, max_size: int = 100, cache_dir: str = "./query_cache"):
        """Initialize query cache.
        
        Args:
            max_size: Maximum number of cached queries (LRU eviction)
            cache_dir: Directory to store persistent cache files
        """
        self.max_size = max_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory LRU cache
        self.memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.total_time_saved = 0.0
        
        self._load_persistent_cache()
    
    def _get_cache_key(self, query: str, vector_weight: float, 
                       keyword_weight: float, top_k: int, 
                       result_type: Optional[str] = None) -> str:
        """Generate a unique cache key for a query."""
        key_str = f"{query}|{vector_weight}|{keyword_weight}|{top_k}|{result_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, vector_weight: float = 0.6, 
            keyword_weight: float = 0.4, top_k: int = 10,
            result_type: Optional[str] = None) -> Optional[Tuple[List[SearchResult], float]]:
        """Retrieve cached query results.
        
        Args:
            query: The search query
            vector_weight: Weight for vector search
            keyword_weight: Weight for keyword search
            top_k: Number of top results
            result_type: Optional filter by type (function/class)
            
        Returns:
            Tuple of (results, cache_age_seconds) if found, None otherwise
        """
        cache_key = self._get_cache_key(query, vector_weight, keyword_weight, top_k, result_type)
        
        if cache_key in self.memory_cache:
            # Move to end (most recently used)
            self.memory_cache.move_to_end(cache_key)
            entry = self.memory_cache[cache_key]
            
            cache_age = time.time() - entry.timestamp
            self.hits += 1
            
            # Reconstruct SearchResult objects
            results = [
                SearchResult(
                    metadata=r['metadata'],
                    combined_score=r['combined_score'],
                    vector_score=r.get('vector_score', 0.0),
                    keyword_score=r.get('keyword_score', 0.0),
                    rank=r.get('rank', i+1)
                )
                for i, r in enumerate(entry.results)
            ]
            
            return (results, cache_age)
        
        self.misses += 1
        return None
    
    def set(self, query: str, results: List, 
            vector_weight: float = 0.6, keyword_weight: float = 0.4, 
            top_k: int = 10, result_type: Optional[str] = None) -> None:
        """Cache query results.
        
        Args:
            query: The search query
            results: Search results to cache (can be SearchResult objects or dicts)
            vector_weight: Weight for vector search
            keyword_weight: Weight for keyword search
            top_k: Number of top results
            result_type: Optional filter by type (function/class)
        """
        cache_key = self._get_cache_key(query, vector_weight, keyword_weight, top_k, result_type)
        
        # Convert SearchResult objects to dictionaries for serialization
        serialized_results = []
        for r in results:
            if isinstance(r, SearchResult):
                serialized_results.append({
                    'metadata': r.metadata,
                    'combined_score': r.combined_score,
                    'vector_score': r.vector_score,
                    'keyword_score': r.keyword_score,
                    'rank': r.rank
                })
            else:
                # Already a dict or dict-like object
                serialized_results.append(r)
        
        entry = CacheEntry(
            query=query,
            results=serialized_results,
            timestamp=time.time(),
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            top_k=top_k,
            result_type=result_type
        )
        
        # Remove oldest item if at capacity
        if len(self.memory_cache) >= self.max_size:
            oldest_key, _ = self.memory_cache.popitem(last=False)
            self._remove_persistent(oldest_key)
        
        # Add new entry
        self.memory_cache[cache_key] = entry
        
        # Persist to disk
        self._save_entry(cache_key, entry)
    
    def clear(self) -> None:
        """Clear all cached queries."""
        self.memory_cache.clear()
        
        # Delete persistent cache files
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        
        self.hits = 0
        self.misses = 0
        self.total_time_saved = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.memory_cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "time_saved_ms": round(self.total_time_saved * 1000, 2),
            "average_time_per_hit_ms": round((self.total_time_saved * 1000 / self.hits) if self.hits > 0 else 0, 2)
        }
    
    def _save_entry(self, cache_key: str, entry: CacheEntry) -> None:
        """Save a cache entry to disk."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache entry: {e}")
    
    def _remove_persistent(self, cache_key: str) -> None:
        """Remove a persistent cache file."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            cache_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to remove cache file: {e}")
    
    def _load_persistent_cache(self) -> None:
        """Load persistent cache from disk on startup."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        entry = CacheEntry.from_dict(data)
                        cache_key = cache_file.stem
                        self.memory_cache[cache_key] = entry
                        
                        # Respect max_size limit when loading
                        if len(self.memory_cache) > self.max_size:
                            self.memory_cache.popitem(last=False)
                
                except Exception as e:
                    print(f"Warning: Failed to load cache entry {cache_file}: {e}")
        
        except Exception as e:
            print(f"Warning: Failed to load persistent cache: {e}")
    
    def export_stats(self, filepath: str) -> None:
        """Export cache statistics to file.
        
        Args:
            filepath: Path to save statistics JSON
        """
        try:
            stats = self.get_stats()
            
            # Add cache contents
            stats["cached_queries"] = [
                {
                    "query": entry.query,
                    "results_count": len(entry.results),
                    "timestamp": entry.timestamp,
                    "age_seconds": round(time.time() - entry.timestamp, 2)
                }
                for entry in self.memory_cache.values()
            ]
            
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to export cache stats: {e}")
