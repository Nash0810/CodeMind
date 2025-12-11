"""
Query history persistence for CodeMind.

Stores and retrieves search history across sessions.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class HistoryEntry:
    """A single search history entry."""
    query: str
    timestamp: float
    results_count: int
    latency_ms: float
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    top_k: int = 10
    result_type: Optional[str] = None
    from_cache: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoryEntry':
        """Create from dictionary."""
        return cls(**data)
    
    @property
    def datetime_str(self) -> str:
        """Get formatted datetime string."""
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.timestamp


class QueryHistory:
    """Manages search history persistence."""
    
    def __init__(self, history_file: str = ".codemind_cache/history.json", max_entries: int = 1000):
        """Initialize query history.
        
        Args:
            history_file: Path to history JSON file
            max_entries: Maximum number of entries to keep
        """
        self.history_file = Path(history_file)
        self.max_entries = max_entries
        self.history: List[HistoryEntry] = []
        
        # Create directory if needed
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing history
        self._load()
    
    def add(self, query: str, results_count: int, latency_ms: float,
            vector_weight: float = 0.6, keyword_weight: float = 0.4,
            top_k: int = 10, result_type: Optional[str] = None,
            from_cache: bool = False) -> None:
        """Add a search to history.
        
        Args:
            query: Search query
            results_count: Number of results returned
            latency_ms: Search latency in milliseconds
            vector_weight: Vector search weight used
            keyword_weight: Keyword search weight used
            top_k: Number of results requested
            result_type: Result type filter if applied
            from_cache: Whether result was from cache
        """
        entry = HistoryEntry(
            query=query,
            timestamp=time.time(),
            results_count=results_count,
            latency_ms=latency_ms,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            top_k=top_k,
            result_type=result_type,
            from_cache=from_cache
        )
        
        self.history.append(entry)
        
        # Trim to max_entries
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]
        
        # Save to disk
        self._save()
    
    def get_recent(self, limit: int = 10) -> List[HistoryEntry]:
        """Get most recent searches.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent history entries
        """
        return self.history[-limit:]
    
    def get_by_query(self, query: str) -> List[HistoryEntry]:
        """Get all entries for a specific query.
        
        Args:
            query: Search query to find
            
        Returns:
            List of matching entries
        """
        return [h for h in self.history if h.query.lower() == query.lower()]
    
    def search_history(self, search_term: str) -> List[HistoryEntry]:
        """Search history for entries matching a term.
        
        Args:
            search_term: Term to search for in queries
            
        Returns:
            List of matching entries
        """
        search_lower = search_term.lower()
        return [h for h in self.history if search_lower in h.query.lower()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get history statistics.
        
        Returns:
            Dictionary with stats
        """
        if not self.history:
            return {
                'total_searches': 0,
                'unique_queries': 0,
                'avg_latency_ms': 0.0,
                'cache_hit_rate_percent': 0.0,
                'total_results_found': 0
            }
        
        unique_queries = len(set(h.query.lower() for h in self.history))
        avg_latency = sum(h.latency_ms for h in self.history) / len(self.history)
        cache_hits = sum(1 for h in self.history if h.from_cache)
        total_results = sum(h.results_count for h in self.history)
        
        return {
            'total_searches': len(self.history),
            'unique_queries': unique_queries,
            'avg_latency_ms': round(avg_latency, 2),
            'cache_hit_rate_percent': round((cache_hits / len(self.history) * 100) if self.history else 0, 2),
            'total_results_found': total_results,
            'oldest_entry': self.history[0].datetime_str if self.history else None,
            'newest_entry': self.history[-1].datetime_str if self.history else None
        }
    
    def clear(self) -> None:
        """Clear all history."""
        self.history.clear()
        self._save()
    
    def export_csv(self, filepath: str) -> None:
        """Export history to CSV file.
        
        Args:
            filepath: Path to save CSV
        """
        import csv
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'Query', 'Timestamp', 'DateTime', 'Results', 'Latency(ms)',
                    'VectorWeight', 'KeywordWeight', 'TopK', 'Type', 'FromCache'
                ])
                
                # Write data
                for entry in self.history:
                    writer.writerow([
                        entry.query,
                        entry.timestamp,
                        entry.datetime_str,
                        entry.results_count,
                        entry.latency_ms,
                        entry.vector_weight,
                        entry.keyword_weight,
                        entry.top_k,
                        entry.result_type or 'all',
                        'Yes' if entry.from_cache else 'No'
                    ])
        
        except Exception as e:
            print(f"Warning: Failed to export history: {e}")
    
    def export_json(self, filepath: str) -> None:
        """Export history to JSON file.
        
        Args:
            filepath: Path to save JSON
        """
        try:
            data = {
                'exported_at': datetime.now().isoformat(),
                'total_entries': len(self.history),
                'statistics': self.get_statistics(),
                'entries': [entry.to_dict() for entry in self.history]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            print(f"Warning: Failed to export history: {e}")
    
    def _save(self) -> None:
        """Save history to disk."""
        try:
            data = {
                'version': 1,
                'saved_at': datetime.now().isoformat(),
                'entries': [entry.to_dict() for entry in self.history]
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            print(f"Warning: Failed to save history: {e}")
    
    def _load(self) -> None:
        """Load history from disk."""
        if not self.history_file.exists():
            return
        
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
            
            entries = data.get('entries', [])
            self.history = [HistoryEntry.from_dict(e) for e in entries]
        
        except Exception as e:
            print(f"Warning: Failed to load history: {e}")
