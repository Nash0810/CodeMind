"""
Interactive TUI for CodeMind search.

Provides an interactive interface with result browsing, filtering, and display.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.align import Align
from rich.text import Text

from src.cli.query import QueryEngine
from src.query_processor import QueryProcessor


class CodeMindTUI:
    """Interactive Text User Interface for CodeMind."""
    
    def __init__(self):
        """Initialize TUI."""
        self.console = Console()
        self.engine = QueryEngine()
        self.processor = QueryProcessor()
        
        self.current_results: List[Dict] = []
        self.current_query: str = ""
        self.running = True
    
    def display_banner(self):
        """Display welcome banner."""
        banner = """
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║                   [bold cyan]CodeMind[/bold cyan] - Code Search Engine                      ║
    ║           Semantic + Keyword Search for Your Codebase         ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
        """
        self.console.print(banner)
        self.console.print("[bold cyan]Commands:[/bold cyan]")
        self.console.print("  search <query>  - Search for code")
        self.console.print("  show <num>      - Show code snippet (1-10)")
        self.console.print("  filter <type>   - Filter by 'function' or 'class'")
        self.console.print("  weight <v> <k>  - Set vector/keyword weights")
        self.console.print("  history         - Show search history")
        self.console.print("  stats           - Show history statistics")
        self.console.print("  help            - Show help")
        self.console.print("  exit            - Exit\n")
        self.console.print("  filter <type>   - Filter by 'function' or 'class'")
        self.console.print("  weight <v> <k>  - Set vector/keyword weights")
        self.console.print("  history         - Show search history")
        self.console.print("  help            - Show help")
        self.console.print("  exit            - Exit\n")
    
    def ensure_indexed(self) -> bool:
        """Ensure code is indexed."""
        if self.engine.vector_store.count() == 0:
            self.console.print("[cyan]Initializing CodeMind...[/cyan]")
            if not self.engine.ensure_indexed():
                return False
        return True
    
    def handle_search(self, query: str, top_k: int = 10):
        """Execute search.
        
        Args:
            query: Search query
            top_k: Number of results
        """
        if not query.strip():
            self.console.print("[yellow]Empty query[/yellow]")
            return
        
        # Process query
        processed_query, strategy = self.processor.process(query)
        self.current_query = query
        
        # Execute search
        self.console.print("[cyan]Searching...[/cyan]")
        
        try:
            results, from_cache, search_time = self.engine.search(
                query,
                top_k=top_k,
                vector_weight=strategy['vector_weight'],
                keyword_weight=strategy['keyword_weight']
            )
            self.current_results = results
            
            # Record in history
            self.engine.query_history.add(
                query=query,
                results_count=len(results),
                latency_ms=search_time * 1000,
                vector_weight=strategy['vector_weight'],
                keyword_weight=strategy['keyword_weight'],
                top_k=top_k,
                result_type=strategy.get('result_type'),
                from_cache=from_cache
            )
            
            # Display results
            self._display_results_table(strategy.get('result_type'))
            
            cache_indicator = "[yellow][cache][/yellow]" if from_cache else ""
            self.console.print(f"\n[green]Found {len(self.current_results)} results in {search_time*1000:.2f}ms[/green] {cache_indicator}")
            
            # Show cache stats
            stats = self.engine.query_cache.get_stats()
            self.console.print(f"[dim]Cache: {stats['cache_size']}/{stats['max_size']} | "
                             f"Hit rate: {stats['hit_rate_percent']}%[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]Search error: {e}[/red]")
    
    def _display_results_table(self, filter_type: Optional[str] = None):
        """Display results in table format.
        
        Args:
            filter_type: Optional filter by type
        """
        results = self.current_results
        
        if filter_type:
            results = [r for r in results 
                      if hasattr(r, 'metadata') and r.metadata.get('type') == filter_type]
        
        if not results:
            self.console.print("[yellow]No results[/yellow]")
            return
        
        table = Table(title=f"Search Results for '{self.current_query}'", 
                     show_header=True, header_style="bold cyan")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Name", style="green", width=20)
        table.add_column("Type", style="magenta", width=10)
        table.add_column("File", style="blue", width=25)
        table.add_column("Score", style="yellow", width=8)
        
        for i, result in enumerate(results[:10], 1):
            metadata = result.metadata if hasattr(result, 'metadata') else result
            
            name = metadata.get('name', 'Unknown')[:20]
            result_type = metadata.get('type', 'unknown')
            file_path = Path(metadata.get('file', 'Unknown')).name[:25]
            score = result.combined_score if hasattr(result, 'combined_score') else 0
            
            table.add_row(
                str(i),
                name,
                result_type,
                file_path,
                f"{score:.3f}"
            )
        
        self.console.print(table)
    
    def handle_show(self, index: int):
        """Show code snippet.
        
        Args:
            index: Result index (1-10)
        """
        if not self.current_results:
            self.console.print("[yellow]No results to show[/yellow]")
            return
        
        if index < 1 or index > len(self.current_results):
            self.console.print(f"[yellow]Invalid index (1-{len(self.current_results)})[/yellow]")
            return
        
        result = self.current_results[index - 1]
        metadata = result.metadata if hasattr(result, 'metadata') else result
        
        name = metadata.get('name', 'Unknown')
        code = metadata.get('code', '')
        docstring = metadata.get('docstring', '')
        result_type = metadata.get('type', 'unknown')
        score = result.combined_score if hasattr(result, 'combined_score') else 0
        
        # Display header
        header = f"{name} ({result_type}) - Score: {score:.3f}"
        self.console.print(Panel(header, style="bold blue"))
        
        # Display docstring if available
        if docstring:
            self.console.print("[bold yellow]Documentation:[/bold yellow]")
            self.console.print(f"  {docstring}\n")
        
        # Display code
        if code:
            try:
                syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                self.console.print(syntax)
            except:
                self.console.print(code)
        else:
            self.console.print("[yellow]No code available[/yellow]")
    
    def handle_history(self):
        """Display search history."""
        recent = self.engine.query_history.get_recent(limit=10)
        
        if not recent:
            self.console.print("[yellow]No search history[/yellow]")
            return
        
        table = Table(title="Search History (Last 10)")
        table.add_column("Query", style="cyan")
        table.add_column("Time", style="green")
        table.add_column("Results", style="yellow")
        table.add_column("Latency", style="magenta")
        table.add_column("Cache", style="blue")
        
        for entry in recent:
            cache_status = "✓" if entry.from_cache else "✗"
            table.add_row(
                entry.query[:30] + ("..." if len(entry.query) > 30 else ""),
                entry.datetime_str,
                str(entry.results_count),
                f"{entry.latency_ms:.1f}ms",
                cache_status
            )
        
        self.console.print(table)
    
    def handle_stats(self):
        """Display history statistics."""
        stats = self.engine.query_history.get_statistics()
        
        if stats['total_searches'] == 0:
            self.console.print("[yellow]No search statistics available[/yellow]")
            return
        
        output = f"""
[bold cyan]Search Statistics:[/bold cyan]
  Total searches: [yellow]{stats['total_searches']}[/yellow]
  Unique queries: [yellow]{stats['unique_queries']}[/yellow]
  Average latency: [yellow]{stats['avg_latency_ms']:.2f}ms[/yellow]
  Cache hit rate: [yellow]{stats['cache_hit_rate_percent']:.1f}%[/yellow]
  Total results: [yellow]{stats['total_results_found']}[/yellow]
  Oldest entry: [dim]{stats['oldest_entry']}[/dim]
  Newest entry: [dim]{stats['newest_entry']}[/dim]
        """
        self.console.print(output)
    
    def display_help(self):
        """Display help information."""
        help_text = """
[bold cyan]CodeMind Search Commands:[/bold cyan]

[bold]search <query>[/bold]
    Execute a semantic/keyword search
    Example: search "functions that fetch data"

[bold]show <num>[/bold]
    Display code snippet from result (1-10)
    Example: show 1

[bold]filter <type>[/bold]
    Filter results by type: function or class
    Example: filter function

[bold]weight <vector> <keyword>[/bold]
    Set search weights (must sum to ~1.0)
    Example: weight 0.7 0.3

[bold]history[/bold]
    Show last 10 searches

[bold]help[/bold]
    Show this help message

[bold]exit[/bold]
    Exit CodeMind

[bold cyan]Tips:[/bold cyan]
• Use natural language: "classes that handle errors"
• Mix keywords: "validate email address"
• Exact matches work: "def authenticate"
        """
        self.console.print(help_text)
    
    def run(self):
        """Run interactive TUI."""
        # Initialize
        if not self.ensure_indexed():
            self.console.print("[red]Failed to initialize CodeMind[/red]")
            return
        
        self.display_banner()
        
        # Main loop
        while self.running:
            try:
                user_input = Prompt.ask("[bold cyan]codemind[/bold cyan]")
                
                if not user_input.strip():
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == 'exit':
                    self.console.print("[cyan]Goodbye![/cyan]")
                    self.running = False
                
                elif command == 'search':
                    self.handle_search(args)
                
                elif command == 'show':
                    try:
                        index = int(args)
                        self.handle_show(index)
                    except ValueError:
                        self.console.print("[yellow]Usage: show <number>[/yellow]")
                
                elif command == 'filter':
                    if args in ['function', 'class']:
                        self._display_results_table(filter_type=args)
                    else:
                        self.console.print("[yellow]Filter type: function or class[/yellow]")
                
                elif command == 'weight':
                    weights = args.split()
                    if len(weights) == 2:
                        try:
                            v_weight = float(weights[0])
                            k_weight = float(weights[1])
                            if abs((v_weight + k_weight) - 1.0) < 0.1:
                                self.console.print(f"[green]Weights set: vector={v_weight}, keyword={k_weight}[/green]")
                            else:
                                self.console.print("[yellow]Warning: weights should sum to 1.0[/yellow]")
                        except ValueError:
                            self.console.print("[yellow]Usage: weight <vector> <keyword>[/yellow]")
                    else:
                        self.console.print("[yellow]Usage: weight <vector> <keyword>[/yellow]")
                
                elif command == 'history':
                    self.handle_history()
                
                elif command == 'stats':
                    self.handle_stats()
                
                elif command == 'help':
                    self.display_help()
                
                else:
                    self.console.print("[yellow]Unknown command. Type 'help' for usage.[/yellow]")
            
            except KeyboardInterrupt:
                self.console.print("\n[cyan]Goodbye![/cyan]")
                self.running = False
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")


def run_tui():
    """Entry point for TUI."""
    tui = CodeMindTUI()
    tui.run()


if __name__ == '__main__':
    run_tui()
