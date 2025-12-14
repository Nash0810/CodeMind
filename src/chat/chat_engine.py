"""
Chat engine orchestrating search, context packing, and LLM generation.

Provides the core Q&A functionality for CodeMind.
"""

from typing import Optional, Iterator, List, Dict, Tuple
from dataclasses import dataclass
import time

from src.search.hybrid_search import HybridSearch, SearchResult
from src.search.context_packer import ContextPacker, CodeSnippet, PackedContext
from src.llm.ollama_client import OllamaClient, GenerationConfig
from src.llm import prompts
from src.query_history import QueryHistory


@dataclass
class ChatMessage:
    """A message in the chat history."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ChatResponse:
    """Response from the chat engine."""
    answer: str
    search_results: List[SearchResult]
    context: Optional[PackedContext]
    latency_ms: float
    model_used: str
    tokens_used: Dict[str, int]  # input_tokens, output_tokens, context_tokens


class ChatEngine:
    """
    Orchestrates search → context → LLM generation.
    
    Pipeline:
    1. Search codebase for relevant code
    2. Pack context using call graph relationships
    3. Generate prompt with context
    4. Call LLM for answer
    5. Return structured response
    """
    
    def __init__(
        self,
        hybrid_search: HybridSearch,
        context_packer: ContextPacker,
        ollama_client: OllamaClient,
        query_history: Optional[QueryHistory] = None
    ):
        """
        Initialize chat engine.
        
        Args:
            hybrid_search: HybridSearch instance for code search
            context_packer: ContextPacker for assembling code context
            ollama_client: OllamaClient for LLM generation
            query_history: Optional QueryHistory for tracking conversations
        """
        self.search = hybrid_search
        self.packer = context_packer
        self.llm = ollama_client
        self.history = query_history or QueryHistory()
        
        self.chat_history: List[ChatMessage] = []
        self.generation_config = GenerationConfig()
    
    def set_model(self, model: str) -> None:
        """Set the LLM model to use."""
        self.generation_config.model = model
    
    def set_generation_config(self, config: GenerationConfig) -> None:
        """Set generation parameters."""
        self.generation_config = config
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        context_depth: int = 2,
        max_context_tokens: int = 6000,
        stream: bool = False,
        query_type: str = "analysis"
    ) -> ChatResponse:
        """
        Answer a question about the codebase.
        
        Args:
            question: User's question
            top_k: Number of search results to retrieve
            context_depth: BFS depth for context packing
            max_context_tokens: Maximum tokens for context
            stream: If True, return generator for streaming response
            query_type: Type of query (analysis, refactoring, debugging, etc)
            
        Returns:
            ChatResponse with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Add to chat history
        self.chat_history.append(ChatMessage(role="user", content=question))
        
        # Step 1: Search
        search_results = self.search.search(question, top_k=top_k)
        if not search_results:
            return ChatResponse(
                answer="No relevant code found in the codebase.",
                search_results=[],
                context=None,
                latency_ms=(time.time() - start_time) * 1000,
                model_used=self.generation_config.model,
                tokens_used={'input': 0, 'output': 0, 'context': 0}
            )
        
        # Step 2: Pack context
        self.packer.token_budget = max_context_tokens
        
        # Use first search result as primary
        primary_result = search_results[0]
        primary_id = f"{primary_result.metadata.get('file')}:{primary_result.metadata.get('line', 0)}:{primary_result.metadata.get('name', 'unknown')}"
        
        # Add other results to snippet cache
        for i, result in enumerate(search_results):
            snippet = CodeSnippet(
                id=f"{result.metadata.get('file')}:{result.metadata.get('line', 0)}:{result.metadata.get('name', 'unknown')}",
                name=result.metadata.get('name', f'snippet_{i}'),
                file=result.metadata.get('file', 'unknown'),
                line=result.metadata.get('line', 0),
                content=result.metadata.get('content', ''),
                type=result.metadata.get('type', 'unknown')
            )
            self.packer.add_snippet(snippet)
        
        # Pack context - build context from search results directly
        context_lines = []
        context_lines.append("# Code Context from Search Results\n")
        
        for i, result in enumerate(search_results[:3], 1):  # Top 3 results
            metadata = result.metadata
            file_path = metadata.get('file', 'unknown')
            name = metadata.get('name', 'unknown')
            code = metadata.get('code', '')
            doc = metadata.get('docstring', '')
            
            context_lines.append(f"## Result {i}: {name}")
            context_lines.append(f"**File**: `{file_path}`")
            if doc:
                context_lines.append(f"**Doc**: {doc[:200]}")
            context_lines.append(f"**Score**: {result.combined_score:.2f}\n")
            if code:
                context_lines.append("```python")
                context_lines.append(code)
                context_lines.append("```\n")
        
        context_text = "\n".join(context_lines) if context_lines else "No context available."
        
        # Step 3: Generate prompt
        prompt_text = prompts.format_query_prompt(question, context_text, query_type)
        
        # Step 4: Generate answer
        if stream:
            # Collect streamed answer into a list, then join
            answer_chunks = list(self._stream_answer(prompt_text))
            answer = ''.join(answer_chunks)
        else:
            answer = self._blocking_answer(prompt_text)
        
        # Add to chat history
        self.chat_history.append(ChatMessage(role="assistant", content=answer))
        
        # Step 5: Build response
        latency_ms = (time.time() - start_time) * 1000
        
        # Estimate tokens
        input_tokens = self.llm.count_tokens(prompt_text)
        output_tokens = self.llm.count_tokens(answer)
        context_tokens = self.llm.count_tokens(context_text) if context_text else 0
        
        response = ChatResponse(
            answer=answer,
            search_results=search_results,
            context=None,  # Not using PackedContext anymore
            latency_ms=latency_ms,
            model_used=self.generation_config.model,
            tokens_used={
                'input': input_tokens,
                'output': output_tokens,
                'context': context_tokens
            }
        )
        
        # Track in history
        self.history.add(question, len(search_results), latency_ms)
        
        return response
    
    def _blocking_answer(self, prompt: str) -> str:
        """Generate answer without streaming."""
        return self.llm.generate(prompt, self.generation_config, stream=False)
    
    def _stream_answer(self, prompt: str) -> Iterator[str]:
        """Generate answer with streaming."""
        return self.llm.generate(prompt, self.generation_config, stream=True)
    
    def chat(
        self,
        message: str,
        system_prompt_type: str = "default",
        stream: bool = False
    ) -> str:
        """
        Chat-style interaction with memory.
        
        Args:
            message: User message
            system_prompt_type: Type of system prompt to use
            stream: If True, return generator for streaming
            
        Returns:
            Assistant response
        """
        # Add user message
        self.chat_history.append(ChatMessage(role="user", content=message))
        
        # Prepare messages for LLM
        system_prompt = prompts.get_system_prompt(system_prompt_type)
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history (limit to last 10 messages)
        for msg in self.chat_history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Generate response
        if stream:
            response = self.llm.chat(messages, self.generation_config, stream=True)
        else:
            response = self.llm.chat(messages, self.generation_config, stream=False)
        
        # Add to history
        if not stream:
            self.chat_history.append(ChatMessage(role="assistant", content=response))
        
        return response
    
    def analyze_code_snippet(self, code: str, context: Optional[str] = None) -> str:
        """
        Analyze a code snippet.
        
        Args:
            code: Code to analyze
            context: Optional additional context
            
        Returns:
            Analysis results
        """
        prompt = f"""Analyze this code snippet:

```python
{code}
```

{f'Additional context: {context}' if context else ''}

Provide:
1. What it does
2. Any potential issues
3. Suggestions for improvement"""
        
        return self.llm.generate(prompt, self.generation_config, stream=False)
    
    def suggest_refactoring(self, code: str) -> str:
        """
        Suggest refactoring for code.
        
        Args:
            code: Code to refactor
            
        Returns:
            Refactoring suggestions
        """
        prompt = prompts.format_refactoring_prompt(code)
        return self.llm.generate(prompt, self.generation_config, stream=False)
    
    def get_chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self.chat_history
    
    def clear_chat_history(self) -> None:
        """Clear chat history."""
        self.chat_history = []
    
    def get_statistics(self) -> Dict[str, any]:
        """Get chat session statistics."""
        if not self.chat_history:
            return {
                'total_messages': 0,
                'user_messages': 0,
                'assistant_messages': 0,
                'session_duration_seconds': 0
            }
        
        first_msg = self.chat_history[0]
        last_msg = self.chat_history[-1]
        
        user_count = sum(1 for m in self.chat_history if m.role == "user")
        assistant_count = sum(1 for m in self.chat_history if m.role == "assistant")
        
        return {
            'total_messages': len(self.chat_history),
            'user_messages': user_count,
            'assistant_messages': assistant_count,
            'session_duration_seconds': last_msg.timestamp - first_msg.timestamp
        }
