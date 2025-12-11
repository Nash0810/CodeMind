"""
Natural language query processing for CodeMind.

Enhances queries with synonyms, query expansion, and intent detection.
"""

from typing import List, Tuple, Dict
from enum import Enum
import re


class QueryIntent(Enum):
    """Types of query intents."""
    FIND_FUNCTION = "find_function"
    FIND_CLASS = "find_class"
    FIND_USAGE = "find_usage"
    FIND_PATTERN = "find_pattern"
    EXPLAIN = "explain"
    GENERIC = "generic"


class QueryProcessor:
    """Process and enhance natural language queries."""
    
    # Query synonyms and expansions
    SYNONYMS = {
        'fetch': ['retrieve', 'get', 'load', 'download', 'pull'],
        'send': ['transmit', 'push', 'post', 'write', 'output'],
        'calculate': ['compute', 'evaluate', 'determine', 'process'],
        'validate': ['check', 'verify', 'test', 'confirm'],
        'parse': ['analyze', 'interpret', 'extract', 'read'],
        'convert': ['transform', 'translate', 'change', 'modify'],
        'iterate': ['loop', 'traverse', 'walk', 'browse'],
        'error': ['exception', 'failure', 'problem', 'issue'],
    }
    
    # Intent patterns
    INTENT_PATTERNS = {
        QueryIntent.FIND_FUNCTION: r'(find|search|locate|get|show|list|where is).*(function|method|def)',
        QueryIntent.FIND_CLASS: r'(find|search|locate|get|show|list).*(class|object|struct|type)',
        QueryIntent.FIND_USAGE: r'(where|find).*(used|called|referenced|called from)',
        QueryIntent.FIND_PATTERN: r'(find|show|list).*(pattern|example|usage|all|similar)',
        QueryIntent.EXPLAIN: r'(explain|what|how does|describe|how\s)'.format(),
    }
    
    def __init__(self):
        """Initialize query processor."""
        self.query_history: List[str] = []
    
    def detect_intent(self, query: str) -> QueryIntent:
        """Detect the intent of a query.
        
        Args:
            query: User query
            
        Returns:
            Detected intent
        """
        query_lower = query.lower()
        
        for intent, pattern in self.INTENT_PATTERNS.items():
            if re.search(pattern, query_lower):
                return intent
        
        return QueryIntent.GENERIC
    
    def expand_query(self, query: str) -> str:
        """Expand query with related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query with synonyms
        """
        expanded = query
        
        for term, synonyms in self.SYNONYMS.items():
            if re.search(r'\b' + term + r'\b', expanded, re.IGNORECASE):
                # Add synonyms as alternative terms
                syn_str = ' OR '.join(synonyms)
                expanded = re.sub(
                    r'\b' + term + r'\b',
                    f'({term} OR {syn_str})',
                    expanded,
                    flags=re.IGNORECASE
                )
        
        return expanded
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract main keywords from query.
        
        Args:
            query: User query
            
        Returns:
            List of keywords
        """
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'is', 'are', 'was', 'were', 'be',
            'that', 'which', 'who', 'what', 'where', 'when', 'why',
            'how', 'find', 'show', 'get', 'list', 'search', 'locate'
        }
        
        # Extract words
        words = re.findall(r'\b[a-z_]+\b', query.lower())
        
        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def normalize_query(self, query: str) -> str:
        """Normalize query for better matching.
        
        Args:
            query: Original query
            
        Returns:
            Normalized query
        """
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove special characters (except underscores)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    def suggest_search_strategy(self, query: str) -> Dict[str, any]:
        """Suggest search strategy based on query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with suggested settings
        """
        intent = self.detect_intent(query)
        keywords = self.extract_keywords(query)
        
        strategy = {
            'intent': intent.value,
            'keywords': keywords,
            'vector_weight': 0.6,
            'keyword_weight': 0.4,
            'top_k': 10,
            'show_code': False
        }
        
        # Adjust strategy based on intent
        if intent == QueryIntent.FIND_FUNCTION:
            strategy['result_type'] = 'function'
            strategy['top_k'] = 5
        elif intent == QueryIntent.FIND_CLASS:
            strategy['result_type'] = 'class'
            strategy['top_k'] = 5
        elif intent == QueryIntent.FIND_USAGE:
            strategy['vector_weight'] = 0.7  # Emphasize semantic search
            strategy['top_k'] = 15
            strategy['show_code'] = True
        elif intent == QueryIntent.FIND_PATTERN:
            strategy['top_k'] = 20
            strategy['show_code'] = True
        elif intent == QueryIntent.EXPLAIN:
            strategy['vector_weight'] = 0.8  # Strong semantic focus
            strategy['show_code'] = True
            strategy['top_k'] = 3
        
        return strategy
    
    def process(self, query: str) -> Tuple[str, Dict]:
        """Process a query and return enhanced version and strategy.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (processed_query, strategy_dict)
        """
        # Add to history
        self.query_history.append(query)
        
        # Normalize
        normalized = self.normalize_query(query)
        
        # Expand (optional - use original for now)
        # expanded = self.expand_query(normalized)
        
        # Suggest strategy
        strategy = self.suggest_search_strategy(query)
        
        return normalized, strategy


if __name__ == '__main__':
    processor = QueryProcessor()
    
    test_queries = [
        "find functions that fetch data",
        "show me all classes",
        "where is this function used?",
        "find similar patterns",
        "explain how this works"
    ]
    
    for q in test_queries:
        processed, strategy = processor.process(q)
        print(f"Query: {q}")
        print(f"Processed: {processed}")
        print(f"Intent: {strategy['intent']}")
        print(f"Keywords: {strategy['keywords']}")
        print()
