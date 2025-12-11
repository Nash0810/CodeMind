"""
Performance optimizations for CodeMind search.

Strategies:
1. Query result caching (already implemented)
2. Embedding cache for repeated queries
3. Batch processing for large result sets
4. Lazy loading of full code content
5. Configurable embedding model size
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
import time


@dataclass
class CachedEmbedding:
    """Cached embedding for a query."""
    query: str
    embedding: List[float]
    timestamp: float
    model_name: str

    def is_stale(self, ttl_seconds: int = 3600) -> bool:
        """Check if embedding cache has expired."""
        return (time.time() - self.timestamp) > ttl_seconds


class EmbeddingCache:
    """LRU cache for query embeddings.
    
    Caches embedding computations to avoid re-embedding identical queries.
    Reduces embedding generation time for repeated semantic searches.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
            ttl_seconds: Time-to-live for cached embeddings (default 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CachedEmbedding] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, query: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[List[float]]:
        """Get cached embedding for query.
        
        Args:
            query: Search query
            model_name: Embedding model name for validation
            
        Returns:
            Cached embedding or None if not found/expired
        """
        key = self._make_key(query, model_name)

        if key not in self.cache:
            self.misses += 1
            return None

        cached = self.cache[key]

        # Check if expired
        if cached.is_stale(self.ttl_seconds):
            del self.cache[key]
            self.misses += 1
            return None

        # Move to end (LRU)
        self.cache.move_to_end(key)
        self.hits += 1
        return cached.embedding

    def set(self, query: str, embedding: List[float], model_name: str = "all-MiniLM-L6-v2"):
        """Cache an embedding.
        
        Args:
            query: Search query
            embedding: Embedding vector
            model_name: Embedding model name
        """
        key = self._make_key(query, model_name)

        # Remove if already exists
        if key in self.cache:
            del self.cache[key]

        # Add to cache
        self.cache[key] = CachedEmbedding(
            query=query,
            embedding=embedding,
            timestamp=time.time(),
            model_name=model_name
        )

        # Evict oldest if over capacity
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size
        }

    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(query: str, model_name: str) -> str:
        """Create cache key from query and model."""
        combined = f"{query}:{model_name}"
        return hashlib.md5(combined.encode()).hexdigest()


class ResultBatcher:
    """Batch-process search results for better performance.
    
    Splits large result sets into manageable batches for:
    - Parallel processing
    - Progressive rendering (show initial results quickly)
    - Memory efficiency
    """

    @staticmethod
    def batch_results(results: List[Dict], batch_size: int = 10) -> List[List[Dict]]:
        """Split results into batches.
        
        Args:
            results: Search results
            batch_size: Number of results per batch
            
        Returns:
            List of result batches
        """
        batches = []
        for i in range(0, len(results), batch_size):
            batches.append(results[i:i + batch_size])
        return batches

    @staticmethod
    def group_by_field(results: List[Dict], field: str) -> Dict[str, List[Dict]]:
        """Group results by field value.
        
        Args:
            results: Search results
            field: Field to group by (e.g., 'file', 'type')
            
        Returns:
            Dictionary mapping field values to result lists
        """
        groups = {}
        for result in results:
            if isinstance(result, dict):
                key = result.get(field, "unknown")
            else:
                key = getattr(result, field, "unknown")

            if key not in groups:
                groups[key] = []
            groups[key].append(result)

        return groups


class QueryNormalizer:
    """Normalize queries for better cache hits.
    
    Improves cache effectiveness by normalizing queries:
    - Remove extra whitespace
    - Normalize punctuation
    - Expand abbreviations
    - Handle synonyms
    """

    # Common abbreviations and their expansions
    ABBREVIATIONS = {
        "fn": "function",
        "func": "function",
        "cls": "class",
        "err": "error",
        "msg": "message",
        "req": "request",
        "resp": "response",
        "db": "database",
        "sql": "SQL",
        "api": "API",
    }

    # Common synonyms for better semantic matching
    SYNONYMS = {
        "find": ["search", "locate", "look for"],
        "get": ["retrieve", "fetch", "obtain"],
        "set": ["update", "modify", "change"],
        "remove": ["delete", "drop", "unset"],
        "error": ["exception", "failure", "bug"],
        "handle": ["manage", "process", "deal with"],
    }

    @staticmethod
    def normalize(query: str) -> str:
        """Normalize query for caching.
        
        Args:
            query: Original query
            
        Returns:
            Normalized query
        """
        # Convert to lowercase
        query = query.lower().strip()

        # Expand abbreviations
        words = query.split()
        expanded = []
        for word in words:
            expanded.append(QueryNormalizer.ABBREVIATIONS.get(word, word))

        return " ".join(expanded)

    @staticmethod
    def get_search_variants(query: str) -> List[str]:
        """Generate search variants for improved recall.
        
        Args:
            query: Original query
            
        Returns:
            List of query variants to search
        """
        variants = [query]  # Include original

        # Add normalized version
        normalized = QueryNormalizer.normalize(query)
        if normalized not in variants:
            variants.append(normalized)

        # Add synonym variants
        words = query.lower().split()
        for word in words:
            if word in QueryNormalizer.SYNONYMS:
                for synonym in QueryNormalizer.SYNONYMS[word]:
                    variant = query.lower().replace(word, synonym)
                    if variant not in variants:
                        variants.append(variant)

        return variants[:3]  # Limit to top 3 variants


class LazyCodeLoader:
    """Lazily load code content to reduce memory usage.
    
    Instead of loading full code content into all result objects,
    store only references and load on demand.
    """

    def __init__(self, code_dir: str):
        """Initialize lazy loader.
        
        Args:
            code_dir: Root directory of code files
        """
        self.code_dir = Path(code_dir)
        self.code_cache: Dict[str, str] = {}  # filepath -> content cache

    def get_code(self, filepath: str, start_line: int = 0, end_line: int = None) -> str:
        """Get code content lazily.
        
        Args:
            filepath: Path to code file
            start_line: Starting line number (0-indexed)
            end_line: Ending line number (inclusive, None for EOF)
            
        Returns:
            Code content or empty string if not found
        """
        # Check cache first
        if filepath in self.code_cache:
            content = self.code_cache[filepath]
        else:
            # Load from disk
            full_path = self.code_dir / filepath
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                self.code_cache[filepath] = content
            except (FileNotFoundError, IsADirectoryError):
                return ""

        # Extract requested lines
        lines = content.split('\n')
        if end_line is None:
            return '\n'.join(lines[start_line:])
        else:
            return '\n'.join(lines[start_line:end_line + 1])

    def clear_cache(self):
        """Clear code cache."""
        self.code_cache.clear()


class VectorSearchOptimizer:
    """Optimize vector search performance.
    
    Strategies:
    - Embedding caching
    - Query expansion for better recall
    - Batch similarity computations
    - Model quantization (future)
    """

    def __init__(self, vector_store, embedding_cache_size: int = 500):
        """Initialize optimizer.
        
        Args:
            vector_store: VectorStore instance
            embedding_cache_size: Size of embedding cache
        """
        self.vector_store = vector_store
        self.embedding_cache = EmbeddingCache(max_size=embedding_cache_size)
        self.query_normalizer = QueryNormalizer()

    def search_with_cache(self, query: str, top_k: int = 10) -> Tuple[List[Dict], bool]:
        """Search with embedding caching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Tuple of (results, from_cache)
        """
        # Try to get cached embedding
        cached_embedding = self.embedding_cache.get(query)

        if cached_embedding:
            # Use cached embedding
            results = self.vector_store.search_by_embedding(cached_embedding, top_k=top_k)
            return (results, True)
        else:
            # Generate embedding and cache it
            query_embedding = self.vector_store.embed_query(query)
            self.embedding_cache.set(query, query_embedding)
            results = self.vector_store.search_by_embedding(query_embedding, top_k=top_k)
            return (results, False)

    def search_with_expansion(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search with query expansion for better recall.
        
        Tries multiple query variants and merges results.
        
        Args:
            query: Original query
            top_k: Results per variant
            
        Returns:
            Combined and deduplicated results
        """
        variants = self.query_normalizer.get_search_variants(query)
        all_results = {}  # id -> result (for deduplication)

        for variant in variants:
            results, _ = self.search_with_cache(variant, top_k=top_k)
            for result in results:
                result_id = result.get('id', id(result))
                if result_id not in all_results:
                    all_results[result_id] = result

        # Return deduplicated results, sorted by score
        return sorted(
            all_results.values(),
            key=lambda r: r.get('score', 0),
            reverse=True
        )[:top_k]

    def get_embedding_stats(self) -> Dict:
        """Get embedding cache statistics."""
        return self.embedding_cache.get_stats()
