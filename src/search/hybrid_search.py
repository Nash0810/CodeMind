"""
Hybrid search combining vector and keyword search.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class SearchResult:
    """Single search result with combined scoring"""
    metadata: Dict[str, Any]
    combined_score: float
    vector_score: float
    keyword_score: float
    rank: int


class HybridSearch:
    """
    Combines vector and keyword search for optimal retrieval.
    
    Strategy:
    - Vector search finds semantically similar code
    - Keyword search finds exact term matches
    - Reranker combines scores with weights
    """
    
    def __init__(self, vector_store, keyword_search):
        """
        Args:
            vector_store: VectorStore instance
            keyword_search: KeywordSearch instance
        """
        self.vector_store = vector_store
        self.keyword_search = keyword_search
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[SearchResult]:
        """
        Performs hybrid search.
        
        Args:
            query: Search query
            top_k: Number of final results
            vector_weight: Weight for vector scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
            
        Returns:
            List of SearchResult objects, ranked by combined score
        """
        start = time.time()
        
        # Get results from both search methods
        vector_start = time.time()
        vector_results = self.vector_store.search(query, top_k=20)
        vector_time = time.time() - vector_start
        
        keyword_start = time.time()
        keyword_results = self.keyword_search.search(query, top_k=20)
        keyword_time = time.time() - keyword_start
        
        # Create a merged results dict
        merged = {}  # id -> {'metadata': ..., 'vector_score': ..., 'keyword_score': ...}
        
        # Add vector results
        for result in vector_results:
            result_id = result['id']
            merged[result_id] = {
                'metadata': result['metadata'],
                'vector_score': result['score'],
                'keyword_score': 0.0
            }
        
        # Add/merge keyword results
        for result in keyword_results:
            # Find matching metadata in merged
            found = False
            for existing_id, existing_data in merged.items():
                if (existing_data['metadata'].get('name') == result['metadata'].get('name') and
                    existing_data['metadata'].get('file') == result['metadata'].get('file')):
                    existing_data['keyword_score'] = result['score']
                    found = True
                    break
            
            # If not found in merged, add it
            if not found:
                result_id = f"kw_{len(merged)}"
                merged[result_id] = {
                    'metadata': result['metadata'],
                    'vector_score': 0.0,
                    'keyword_score': result['score']
                }
        
        # Calculate combined scores and create SearchResults
        search_results = []
        for result_id, data in merged.items():
            vector_score = data['vector_score']
            keyword_score = data['keyword_score']
            
            # Normalize scores to 0-1 range
            vector_norm = min(vector_score, 1.0)  # Already 0-1
            keyword_norm = min(keyword_score / 20.0, 1.0)  # BM25 can exceed 1
            
            # Combined score
            combined = (vector_weight * vector_norm) + (keyword_weight * keyword_norm)
            
            search_results.append({
                'metadata': data['metadata'],
                'combined_score': combined,
                'vector_score': vector_score,
                'keyword_score': keyword_score
            })
        
        # Sort by combined score
        search_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Create SearchResult objects with ranks
        results = [
            SearchResult(
                metadata=r['metadata'],
                combined_score=r['combined_score'],
                vector_score=r['vector_score'],
                keyword_score=r['keyword_score'],
                rank=i + 1
            )
            for i, r in enumerate(search_results[:top_k])
        ]
        
        elapsed = time.time() - start
        
        print(f"\n[Hybrid Search]")
        print(f"  Vector search: {vector_time:.3f}s ({len(vector_results)} results)")
        print(f"  Keyword search: {keyword_time:.3f}s ({len(keyword_results)} results)")
        print(f"  Total: {elapsed:.3f}s")
        print(f"  Combined: {len(results)} results (top {top_k})")
        
        return results
