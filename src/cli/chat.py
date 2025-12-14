"""
Chat command for CodeMind CLI.

Provides interactive chat with the codebase using LLM integration.
"""

import sys
import click
import os
from pathlib import Path
from typing import Optional

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.parser.ast_parser import parse_file
from src.parser.directory_walker import walk_directory
from src.search.hybrid_search import HybridSearch
from src.indexing.vector_store import VectorStore
from src.indexing.keyword_search import KeywordSearch
from src.search.context_packer import ContextPacker
from src.llm.ollama_client import OllamaClient, GenerationConfig
from src.chat.chat_engine import ChatEngine
from src.query_history import QueryHistory


@click.command()
@click.argument('query', required=False)
@click.option('--codebase', '-c', type=click.Path(exists=True), help='Path to codebase')
@click.option('--interactive', '-i', is_flag=True, help='Interactive chat mode')
@click.option('--model', '-m', default='neural-chat', help='Ollama model to use')
@click.option('--top-k', '-k', default=5, type=int, help='Number of search results')
@click.option('--context-depth', '-d', default=2, type=int, help='Call graph depth for context')
@click.option('--stream', '-s', is_flag=True, help='Stream response')
@click.option('--no-context', is_flag=True, help='Skip context packing (faster)')
def chat_command(
    query: Optional[str],
    codebase: Optional[str],
    interactive: bool,
    model: str,
    top_k: int,
    context_depth: int,
    stream: bool,
    no_context: bool
):
    """
    Interactive chat with codebase using LLM.
    
    Ask questions about your code and get intelligent answers with full context.
    
    Examples:
        codemind chat "What does the authentication system do?"
        codemind chat -i  # Interactive mode
        codemind chat -c /path/to/code "Explain the database layer"
    """
    
    # Get codebase path
    if not codebase:
        codebase = os.getcwd()
    
    codebase_path = Path(codebase)
    
    # Initialize components
    try:
        click.echo("[*] Initializing CodeMind...")
        
        # Parse codebase
        click.echo(f"[*] Parsing codebase: {codebase_path}")
        parsed_files = list(walk_directory(str(codebase_path)))
        if not parsed_files:
            click.echo("[!] No Python files found in codebase")
            return
        
        click.echo(f"[+] Parsed {len(parsed_files)} files")
        
        # Initialize vector store and search
        click.echo("[*] Building search index...")
        vector_store = VectorStore(collection_name="codemind_chat")
        keyword_search = KeywordSearch()
        
        # Index all code blocks
        vector_store.index_code_blocks(parsed_files)
        
        click.echo("[+] Search index ready")
        
        # Initialize hybrid search
        hybrid_search = HybridSearch(vector_store, keyword_search)
        
        # Initialize context packer
        context_packer = ContextPacker(token_budget=6000)
        
        # Initialize LLM client
        click.echo("[*] Initializing LLM...")
        ollama = OllamaClient()
        
        if not ollama.is_available():
            click.echo("[!] Ollama server not running. Start with: ollama serve")
            return
        
        available_models = ollama.list_models()
        if model not in available_models:
            click.echo(f"[!] Model '{model}' not available. Available: {', '.join(available_models)}")
            if available_models:
                model = available_models[0]
                click.echo(f"Using: {model}")
            else:
                click.echo("No models available. Pull a model with: ollama pull neural-chat")
                return
        
        click.echo(f"[+] LLM ready ({model})")
        
        # Initialize chat engine
        query_history = QueryHistory()
        chat_engine = ChatEngine(hybrid_search, context_packer, ollama, query_history)
        chat_engine.set_model(model)
        
        click.echo("[+] Chat engine initialized\n")
        
    except Exception as e:
        click.echo(f"[!] Initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Handle single query
    if query and not interactive:
        _handle_single_query(chat_engine, query, top_k, context_depth, stream, no_context)
        return
    
    # Interactive mode
    _interactive_chat(chat_engine, top_k, context_depth, stream, no_context)


def _handle_single_query(
    chat_engine: ChatEngine,
    query: str,
    top_k: int,
    context_depth: int,
    stream: bool,
    no_context: bool
):
    """Handle a single query."""
    click.echo(f"[Q] {query}\n")
    
    try:
        # Get answer
        if no_context:
            # Direct LLM response without context
            response_text = chat_engine.llm.generate(query, chat_engine.generation_config, stream=False)
            if stream:
                for chunk in chat_engine.llm.generate(query, chat_engine.generation_config, stream=True):
                    click.echo(chunk, nl=False)
            else:
                click.echo(response_text)
        else:
            # Full chat with context
            response = chat_engine.answer_question(
                query,
                top_k=top_k,
                context_depth=context_depth,
                stream=stream
            )
            
            if stream:
                # Streaming response
                for chunk in response.answer:
                    click.echo(chunk, nl=False)
                click.echo()
            else:
                # Full response
                click.echo(response.answer)
            
            # Show metadata
            click.echo("\n" + "="*60)
            click.echo("[INFO] Response Metadata:")
            click.echo(f"  Model: {response.model_used}")
            click.echo(f"  Latency: {response.latency_ms:.1f}ms")
            click.echo(f"  Tokens: {response.tokens_used['input']} input, {response.tokens_used['output']} output")
            click.echo(f"  Context tokens: {response.tokens_used['context']}")
            click.echo(f"  Search results: {len(response.search_results)}")
            
            # Show sources
            if response.search_results:
                click.echo("\n[SOURCES]")
                for i, result in enumerate(response.search_results, 1):
                    file = result.metadata.get('file', 'unknown')
                    line = result.metadata.get('line', '?')
                    name = result.metadata.get('name', 'unknown')
                    score = result.combined_score
                    click.echo(f"  {i}. {file}:{line} ({name}) - score: {score:.2f}")
    
    except Exception as e:
        click.echo(f"[!] Error: {str(e)}")


def _interactive_chat(
    chat_engine: ChatEngine,
    top_k: int,
    context_depth: int,
    stream: bool,
    no_context: bool
):
    """Interactive chat loop."""
    click.echo("[CHAT] CodeMind Chat - Type 'exit' to quit, 'help' for commands\n")
    click.echo("Commands:")
    click.echo("  exit, quit, q     - Exit chat")
    click.echo("  clear              - Clear chat history")
    click.echo("  history            - Show chat history")
    click.echo("  stats              - Show session statistics")
    click.echo("  model <name>       - Switch model")
    click.echo("")
    
    while True:
        try:
            # Get user input
            user_input = click.prompt("You").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                click.echo("Goodbye!")
                break
            
            elif user_input.lower() == 'clear':
                chat_engine.clear_chat_history()
                click.echo("✓ Chat history cleared")
                continue
            
            elif user_input.lower() == 'history':
                history = chat_engine.get_chat_history()
                if not history:
                    click.echo("(No chat history)")
                else:
                    for msg in history:
                        role = "You" if msg.role == "user" else "CodeMind"
                        click.echo(f"{role}: {msg.content[:100]}...")
                continue
            
            elif user_input.lower() == 'stats':
                stats = chat_engine.get_statistics()
                click.echo(f"Messages: {stats['total_messages']} ({stats['user_messages']} user, {stats['assistant_messages']} assistant)")
                click.echo(f"Duration: {stats['session_duration_seconds']:.1f}s")
                continue
            
            elif user_input.lower().startswith('model '):
                new_model = user_input[6:].strip()
                chat_engine.set_model(new_model)
                click.echo(f"✓ Switched to model: {new_model}")
                continue
            
            elif user_input.lower() == 'help':
                click.echo("Commands:")
                click.echo("  exit, quit, q     - Exit chat")
                click.echo("  clear              - Clear chat history")
                click.echo("  history            - Show chat history")
                click.echo("  stats              - Show session statistics")
                click.echo("  model <name>       - Switch model")
                continue
            
            # Regular query
            click.echo()
            try:
                if no_context:
                    # Simple chat without code context
                    if stream:
                        click.echo("CodeMind: ", nl=False)
                        for chunk in chat_engine.chat(user_input, stream=True):
                            click.echo(chunk, nl=False)
                        click.echo()
                    else:
                        response = chat_engine.chat(user_input, stream=False)
                        click.echo(f"CodeMind: {response}")
                else:
                    # Full chat with context
                    response = chat_engine.answer_question(
                        user_input,
                        top_k=top_k,
                        context_depth=context_depth,
                        stream=stream
                    )
                    
                    if stream:
                        click.echo("CodeMind: ", nl=False)
                        for chunk in response.answer:
                            click.echo(chunk, nl=False)
                        click.echo()
                    else:
                        click.echo(f"CodeMind: {response.answer}")
                    
                    # Show brief metadata
                    click.echo(f"[{response.latency_ms:.0f}ms, {response.tokens_used['output']} tokens, {len(response.search_results)} sources]")
                
                click.echo()
            
            except Exception as e:
                click.echo(f"Error: {str(e)}\n")
        
        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            break
        except EOFError:
            click.echo("\nGoodbye!")
            break
