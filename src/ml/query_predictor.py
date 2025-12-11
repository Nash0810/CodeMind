"""
ML Query Prediction Module

Predicts next user query, suggests refinements, and learns from search patterns.
Uses n-grams, sequence analysis, and embeddings similarity.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class PredictionResult:
    """Result from query prediction."""
    predicted_query: str
    confidence: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class QuerySuggestion:
    """Suggestion for query refinement."""
    original_query: str
    suggested_query: str
    suggestion_type: str  # "expand", "clarify", "simplify", "related"
    reasoning: str
    confidence: float


@dataclass
class QueryPattern:
    """Pattern in user search behavior."""
    pattern_type: str  # "sequence", "parallel", "repetition", "theme"
    queries: List[str]
    frequency: int
    success_rate: float
    avg_results: float


class QuerySequenceAnalyzer:
    """Analyze sequences of user queries."""
    
    def __init__(self, window_size: int = 5):
        """Initialize sequence analyzer.
        
        Args:
            window_size: Number of recent queries to consider
        """
        self.window_size = window_size
        self.sequences = defaultdict(int)  # Map n-gram to count
        self.sequence_outcomes = defaultdict(list)  # Map n-gram to results
    
    def record_query(self, query: str, success: bool, result_count: int):
        """Record a query and its outcome.
        
        Args:
            query: Search query
            success: Whether search succeeded
            result_count: Number of results returned
        """
        # Store for later analysis
        pass
    
    def get_next_query_candidates(self,
                                  query_history: List[str],
                                  limit: int = 5) -> List[Tuple[str, float]]:
        """Get candidates for next query based on history.
        
        Args:
            query_history: Recent queries in order
            limit: Maximum candidates to return
            
        Returns:
            List of (query, probability) tuples
        """
        if not query_history or len(query_history) == 0:
            return []
        
        # Build n-grams from history
        candidates = Counter()
        
        # Look for similar sequences
        for seq_len in range(1, min(len(query_history) + 1, self.window_size + 1)):
            seq_key = tuple(query_history[-seq_len:])
            # In real implementation, would look up in sequences dict
        
        # Return top candidates with probabilities
        total = sum(candidates.values()) if candidates else 1
        return [(q, c/total) for q, c in candidates.most_common(limit)]


class QueryEmbeddingPredictor:
    """Use embeddings to predict and suggest queries."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding-based predictor.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.query_embeddings = {}  # Cache query embeddings
        self.query_history = []
    
    def add_query(self, query: str, embedding: Optional[np.ndarray] = None):
        """Add query to history.
        
        Args:
            query: Search query
            embedding: Pre-computed embedding (optional)
        """
        if embedding is None:
            embedding = self.model.encode(query)
        
        self.query_embeddings[query] = embedding
        self.query_history.append(query)
        
        # Keep recent queries (last 1000)
        if len(self.query_history) > 1000:
            old_query = self.query_history.pop(0)
            del self.query_embeddings[old_query]
    
    def find_similar_queries(self,
                            query: str,
                            limit: int = 5,
                            threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find similar queries from history.
        
        Args:
            query: Query to find similar to
            limit: Maximum results
            threshold: Minimum similarity score
            
        Returns:
            List of (query, similarity) tuples
        """
        query_embedding = self.model.encode(query)
        similarities = []
        
        for hist_query, hist_embedding in self.query_embeddings.items():
            if hist_query == query:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, hist_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(hist_embedding) + 1e-8
            )
            
            if similarity >= threshold:
                similarities.append((hist_query, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def suggest_related_queries(self,
                               query: str,
                               limit: int = 3) -> List[QuerySuggestion]:
        """Suggest related queries based on embeddings.
        
        Args:
            query: Current query
            limit: Maximum suggestions
            
        Returns:
            List of QuerySuggestion objects
        """
        similar = self.find_similar_queries(query, limit=limit*2, threshold=0.5)
        suggestions = []
        
        for similar_query, similarity in similar:
            suggestion = QuerySuggestion(
                original_query=query,
                suggested_query=similar_query,
                suggestion_type="related",
                reasoning=f"Similar to your previous search: '{similar_query}'",
                confidence=similarity
            )
            suggestions.append(suggestion)
        
        return suggestions[:limit]


class QueryExpander:
    """Expand queries based on patterns and context."""
    
    def __init__(self):
        """Initialize query expander."""
        self.expansions = {
            'find': ['locate', 'search for', 'look for'],
            'function': ['method', 'def', 'subroutine', 'procedure'],
            'class': ['type', 'struct', 'object', 'interface'],
            'usage': ['usage', 'calls', 'references', 'implementations'],
            'pattern': ['template', 'design pattern', 'example'],
        }
    
    def expand_query(self, query: str, max_variants: int = 3) -> List[str]:
        """Generate query variants through expansion.
        
        Args:
            query: Original query
            max_variants: Maximum variants to generate
            
        Returns:
            List of query variants
        """
        variants = [query]  # Include original
        
        query_lower = query.lower()
        
        # Generate expansions based on keywords
        for keyword, synonyms in self.expansions.items():
            if keyword in query_lower:
                for synonym in synonyms[:1]:  # Use first synonym
                    variant = query_lower.replace(keyword, synonym)
                    if variant not in variants:
                        variants.append(variant)
                        if len(variants) >= max_variants + 1:
                            break
        
        return variants[:max_variants + 1]
    
    def clarify_vague_query(self, query: str) -> List[QuerySuggestion]:
        """Suggest clarifications for vague queries.
        
        Args:
            query: Potentially vague query
            
        Returns:
            List of clarification suggestions
        """
        suggestions = []
        query_lower = query.lower()
        
        # Check for vague terms
        vague_patterns = {
            'it': 'Please be more specific about the item',
            'thing': 'What specific thing are you looking for?',
            'stuff': 'Could you clarify what you mean?',
        }
        
        for vague_term, clarification in vague_patterns.items():
            if vague_term in query_lower:
                suggestions.append(QuerySuggestion(
                    original_query=query,
                    suggested_query=query,  # User should refine
                    suggestion_type="clarify",
                    reasoning=clarification,
                    confidence=0.8
                ))
        
        return suggestions


class QueryPredictor:
    """Main query prediction system."""
    
    def __init__(self,
                 history_file: str = ".codemind_cache/ml_query_history.json",
                 model_name: str = "all-MiniLM-L6-v2"):
        """Initialize query predictor.
        
        Args:
            history_file: File to persist training data
            model_name: Embedding model name
        """
        self.history_file = Path(history_file)
        self.sequence_analyzer = QuerySequenceAnalyzer()
        self.embedding_predictor = QueryEmbeddingPredictor(model_name)
        self.expander = QueryExpander()
        self.training_data = self._load_training_data()
    
    def train(self, query: str, success: bool, result_count: int):
        """Train predictor on user query.
        
        Args:
            query: Search query
            success: Whether search succeeded
            result_count: Number of results
        """
        # Record in sequence analyzer
        self.sequence_analyzer.record_query(query, success, result_count)
        
        # Add to embedding predictor
        self.embedding_predictor.add_query(query)
        
        # Record in training data
        self.training_data.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'success': success,
            'result_count': result_count
        })
        
        # Keep recent data (last 10000 records)
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-10000:]
        
        self._save_training_data()
    
    def predict_next_query(self,
                          query_history: List[str],
                          limit: int = 3) -> List[PredictionResult]:
        """Predict next user query.
        
        Args:
            query_history: Recent queries in order
            limit: Maximum predictions
            
        Returns:
            List of PredictionResult objects
        """
        if not query_history:
            return []
        
        current_query = query_history[-1]
        predictions = []
        
        # Get sequence-based predictions
        seq_candidates = self.sequence_analyzer.get_next_query_candidates(
            query_history, limit=limit
        )
        
        # Get embedding-based predictions (similar queries)
        similar = self.embedding_predictor.find_similar_queries(
            current_query, limit=limit, threshold=0.6
        )
        
        # Combine and rank
        all_candidates = {}
        
        for query, prob in seq_candidates:
            all_candidates[query] = all_candidates.get(query, 0) + prob * 0.6
        
        for query, similarity in similar:
            all_candidates[query] = all_candidates.get(query, 0) + similarity * 0.4
        
        # Create results
        sorted_candidates = sorted(
            all_candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (query, score) in enumerate(sorted_candidates[:limit]):
            alternatives = sorted_candidates[i+1:limit+1]
            predictions.append(PredictionResult(
                predicted_query=query,
                confidence=min(score, 1.0),
                alternatives=alternatives,
                reasoning=f"Based on your recent search patterns"
            ))
        
        return predictions
    
    def suggest_query_refinement(self, query: str, results_count: int = 0) -> List[QuerySuggestion]:
        """Suggest ways to refine query.
        
        Args:
            query: Current query
            results_count: Number of results returned
            
        Returns:
            List of QuerySuggestion objects
        """
        suggestions = []
        
        # Check for vague queries
        vague_suggestions = self.expander.clarify_vague_query(query)
        suggestions.extend(vague_suggestions)
        
        # Check if results are low (suggest expansion)
        if results_count == 0:
            expanded = self.expander.expand_query(query, max_variants=2)
            for expanded_query in expanded[1:]:  # Skip original
                suggestions.append(QuerySuggestion(
                    original_query=query,
                    suggested_query=expanded_query,
                    suggestion_type="expand",
                    reasoning="No results found. Try a broader search",
                    confidence=0.7
                ))
        elif results_count < 3:
            # Low results, suggest expansion
            expanded = self.expander.expand_query(query, max_variants=1)
            if len(expanded) > 1:
                suggestions.append(QuerySuggestion(
                    original_query=query,
                    suggested_query=expanded[1],
                    suggestion_type="expand",
                    reasoning=f"Only {results_count} results found. Try expanding your search",
                    confidence=0.6
                ))
        
        # Suggest related queries from embedding predictor
        related = self.embedding_predictor.suggest_related_queries(
            query, limit=2
        )
        suggestions.extend(related)
        
        return suggestions
    
    def get_search_patterns(self) -> List[QueryPattern]:
        """Analyze search patterns from training data.
        
        Returns:
            List of QueryPattern objects
        """
        patterns = []
        
        if not self.training_data:
            return patterns
        
        # Analyze query frequencies and success rates
        query_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'results': []})
        
        for entry in self.training_data:
            query = entry['query']
            query_stats[query]['total'] += 1
            if entry['success']:
                query_stats[query]['success'] += 1
            query_stats[query]['results'].append(entry['result_count'])
        
        # Create patterns for frequent queries
        for query, stats in query_stats.items():
            if stats['total'] >= 3:  # Only patterns with 3+ occurrences
                pattern = QueryPattern(
                    pattern_type="repetition",
                    queries=[query],
                    frequency=stats['total'],
                    success_rate=stats['success'] / stats['total'],
                    avg_results=statistics.mean(stats['results'])
                )
                patterns.append(pattern)
        
        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns
    
    def export_model(self, filepath: str):
        """Export trained model data.
        
        Args:
            filepath: Output file path
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'training_records': len(self.training_data),
            'query_embeddings_count': len(self.embedding_predictor.query_embeddings),
            'patterns': [asdict(p) for p in self.get_search_patterns()],
            'sample_data': self.training_data[-100:]  # Last 100 records
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    # Helper methods
    
    def _load_training_data(self) -> List[Dict]:
        """Load training data from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _save_training_data(self):
        """Save training data to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.training_data, f, indent=2)


# ==============================================================================
# CLI Commands
# ==============================================================================

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


console = Console()


@click.group()
def ml_cli():
    """ML query prediction commands."""
    pass


@ml_cli.command()
@click.argument('query')
@click.option('-l', '--limit', default=3, help='Number of predictions')
def predict(query, limit):
    """Predict next query based on current query."""
    try:
        predictor = QueryPredictor()
        
        # For demo, use query as single history
        predictions = predictor.predict_next_query([query], limit=limit)
        
        if predictions:
            console.print(Panel("[bold]Query Predictions[/bold]"))
            for i, pred in enumerate(predictions, 1):
                console.print(f"\n{i}. [cyan]{pred.predicted_query}[/cyan]")
                console.print(f"   Confidence: {pred.confidence:.1%}")
                if pred.reasoning:
                    console.print(f"   {pred.reasoning}")
        else:
            console.print("[yellow]No predictions available[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@ml_cli.command()
@click.argument('query')
def suggest(query):
    """Suggest query refinements."""
    try:
        predictor = QueryPredictor()
        suggestions = predictor.suggest_query_refinement(query)
        
        if suggestions:
            console.print(Panel("[bold]Query Suggestions[/bold]"))
            
            table = Table()
            table.add_column("Type", style="cyan")
            table.add_column("Suggestion", style="green")
            table.add_column("Reason", style="yellow")
            
            for sugg in suggestions[:5]:
                table.add_row(
                    sugg.suggestion_type,
                    sugg.suggested_query,
                    sugg.reasoning[:40] + "..." if len(sugg.reasoning) > 40 else sugg.reasoning
                )
            
            console.print(table)
        else:
            console.print("[green]Query looks good![/green]")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@ml_cli.command()
def patterns():
    """Show detected search patterns."""
    try:
        predictor = QueryPredictor()
        patterns = predictor.get_search_patterns()
        
        if patterns:
            console.print(Panel("[bold]Search Patterns[/bold]"))
            
            table = Table()
            table.add_column("Pattern", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Frequency", style="green")
            table.add_column("Success %", style="yellow")
            
            for pattern in patterns[:10]:
                table.add_row(
                    pattern.queries[0][:40],
                    pattern.pattern_type,
                    str(pattern.frequency),
                    f"{pattern.success_rate:.0%}"
                )
            
            console.print(table)
        else:
            console.print("[yellow]No patterns found yet[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@ml_cli.command()
@click.option('-o', '--output', required=True, help='Output file')
def export(output):
    """Export model data."""
    try:
        predictor = QueryPredictor()
        predictor.export_model(output)
        console.print(f"[green]✓ Model exported to {output}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == '__main__':
    ml_cli()
