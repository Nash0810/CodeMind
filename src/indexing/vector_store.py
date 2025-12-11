"""
Vector database for semantic code search using ChromaDB.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path
import time


class VectorStore:
    """
    Vector database for semantic code search using ChromaDB.
    Uses sentence-transformers for code embeddings.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "code_embeddings"):
        """
        Initialize ChromaDB client and embedding model.
        
        Args:
            persist_directory: Where to store the database
            collection_name: Name of the collection
        """
        print(f"Initializing VectorStore at: {persist_directory}")
        
        self.persist_dir = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Create ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Load embedding model (all-MiniLM for general code)
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("[OK] Embedding model loaded")
    
    def index_code_blocks(self, parsed_files: List[Dict[str, Any]]):
        """
        Indexes code blocks from parsed repository data.
        
        Args:
            parsed_files: List of file metadata dicts from parser
        """
        print("\n" + "="*60)
        print("VECTOR INDEXING")
        print("="*60)
        
        documents = []
        metadatas = []
        ids = []
        
        idx = 0
        
        # Process each file
        for file_data in parsed_files:
            file_path = file_data.get('file') or file_data.get('file_path')
            
            # Index functions
            for func in file_data.get('functions', []):
                doc = self._format_code_block(func, file_path, 'function')
                documents.append(doc)
                
                metadatas.append({
                    'file': file_path,
                    'name': func['name'],
                    'line_start': func['line_start'],
                    'line_end': func['line_end'],
                    'type': 'function',
                    'docstring': (func.get('docstring') or '')[:200],  # Truncate for metadata
                    'code': (func.get('code') or '')[:500]  # Store snippet in metadata
                })
                
                ids.append(f"func_{idx}")
                idx += 1
            
            # Index classes (including their methods)
            for cls in file_data.get('classes', []):
                doc = self._format_code_block(cls, file_path, 'class')
                documents.append(doc)
                
                metadatas.append({
                    'file': file_path,
                    'name': cls['name'],
                    'line_start': cls['line_start'],
                    'line_end': cls['line_end'],
                    'type': 'class',
                    'docstring': (cls.get('docstring') or '')[:200],
                    'code': ''  # Classes may not have code
                })
                
                ids.append(f"class_{idx}")
                idx += 1
        
        print(f"\nTotal code blocks to index: {len(documents)}")
        
        if len(documents) == 0:
            print("⚠ No code blocks to index")
            return
        
        # Batch embed (this is the slow part)
        print("Generating embeddings...")
        start = time.time()
        
        embeddings = self.embedder.encode(
            documents,
            show_progress_bar=True,
            batch_size=32  # Adjust based on available RAM
        )
        
        embed_time = time.time() - start
        print(f"[OK] Embeddings generated in {embed_time:.2f}s")
        print(f"  Rate: {len(documents)/embed_time:.1f} blocks/sec")
        
        # Add to ChromaDB
        print("Storing in ChromaDB...")
        start = time.time()
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        store_time = time.time() - start
        print(f"[OK] Stored in {store_time:.2f}s")
        
        print(f"\n[SUCCESS] Indexed {len(documents)} code blocks")
    
    def _format_code_block(self, code_unit: dict, file_path: str, block_type: str) -> str:
        """
        Formats a code block for embedding.
        
        Strategy: Include file path, name, docstring, and code.
        The embedding model will learn to associate these.
        """
        parts = []
        
        # Add context
        parts.append(f"File: {file_path}")
        parts.append(f"Type: {block_type}")
        parts.append(f"Name: {code_unit['name']}")
        
        # Add docstring if present
        if code_unit.get('docstring'):
            parts.append(f"Description: {code_unit['docstring']}")
        
        # Add code if present (classes may not have it)
        if code_unit.get('code'):
            parts.append(f"Code:\n{code_unit['code']}")
        
        # For classes, add base classes info
        if block_type == 'class' and code_unit.get('base_classes'):
            parts.append(f"Bases: {', '.join(code_unit['base_classes'])}")
        
        return "\n".join(parts)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search.
        
        Args:
            query: Search query (natural language or code snippet)
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        # Embed the query
        query_embedding = self.embedder.encode([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Handle empty results
        if not results['ids'] or len(results['ids'][0]) == 0:
            return []
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'score': 1 - results['distances'][0][i]  # Convert cosine distance to similarity
            })
        
        return formatted
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Search query
            
        Returns:
            Embedding vector as list
        """
        embedding = self.embedder.encode([query])[0]
        return embedding.tolist()
    
    def search_by_embedding(self, embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using a pre-computed embedding.
        
        Useful for optimization strategies that cache embeddings.
        
        Args:
            embedding: Pre-computed embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        
        # Handle empty results
        if not results['ids'] or len(results['ids'][0]) == 0:
            return []
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'score': 1 - results['distances'][0][i]
            })
        
        return formatted
    
    def count(self) -> int:
        """Returns number of indexed blocks"""
        return self.collection.count()
    
    def clear(self):
        """Clears the collection"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
