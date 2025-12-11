"""
Keyword-based search using BM25 algorithm.
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import re


class KeywordSearch:
    """
    Keyword-based search using BM25 algorithm.
    Complements vector search for exact term matching.
    """
    
    def __init__(self):
        """Initialize keyword search index."""
        self.corpus = []  # List of tokenized documents
        self.metadata = []  # Corresponding metadata
        self.bm25 = None
    
    def index_code_blocks(self, parsed_files: List[Dict[str, Any]]):
        """
        Builds BM25 index from parsed data.
        
        Args:
            parsed_files: List of file metadata dicts from parser
        """
        print("\n" + "="*60)
        print("KEYWORD INDEXING (BM25)")
        print("="*60)
        
        for file_data in parsed_files:
            file_path = file_data.get('file') or file_data.get('file_path')
            
            # Index functions
            for func in file_data.get('functions', []):
                tokens = self._tokenize_code_block(func)
                self.corpus.append(tokens)
                
                self.metadata.append({
                    'file': file_path,
                    'name': func['name'],
                    'line_start': func['line_start'],
                    'line_end': func['line_end'],
                    'type': 'function',
                    'code': func['code']
                })
            
            # Index classes
            for cls in file_data.get('classes', []):
                tokens = self._tokenize_code_block(cls)
                self.corpus.append(tokens)
                
                self.metadata.append({
                    'file': file_path,
                    'name': cls['name'],
                    'line_start': cls['line_start'],
                    'line_end': cls['line_end'],
                    'type': 'class',
                    'code': ''  # Classes don't have code field
                })
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.corpus)
        print(f"[SUCCESS] Built BM25 index with {len(self.corpus)} documents")
    
    def _tokenize_code_block(self, code_unit: dict) -> List[str]:
        """
        Tokenizes code for BM25 indexing.
        
        Strategy:
        - Extract function/class name (high weight)
        - Extract all identifiers from code
        - Extract words from docstring
        - Keep lowercase for case-insensitive matching
        """
        tokens = []
        
        # Add name multiple times (boost importance)
        name = code_unit['name']
        tokens.extend([name.lower()] * 3)  # 3x weight for name
        
        # Extract identifiers from code (if present)
        code = code_unit.get('code', '')
        if code:
            identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
            tokens.extend([id.lower() for id in identifiers])
        
        # Add docstring words
        if code_unit.get('docstring'):
            doc_words = re.findall(r'\b\w+\b', code_unit['docstring'].lower())
            tokens.extend(doc_words)
        
        return tokens
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs BM25 keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with BM25 scores
        """
        if not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    'metadata': self.metadata[idx],
                    'score': float(scores[idx]),
                    'matched_tokens': query_tokens
                })
        
        return results
