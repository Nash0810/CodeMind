"""
Prompts for CodeMind LLM integration.

Carefully crafted system prompts and templates to guide LLM behavior.
"""

# System prompt for code analysis
SYSTEM_PROMPT = """You are CodeMind, an expert code analysis assistant. You help developers understand and work with codebases.

Your capabilities:
1. Analyze code snippets and explain their functionality
2. Identify potential bugs, security issues, and code smells
3. Suggest improvements and refactoring opportunities
4. Answer questions about code relationships and dependencies
5. Generate documentation for code

When answering questions:
- Be concise but thorough
- Use code examples when helpful
- Explain the "why" not just the "what"
- Consider performance and maintainability
- Point out any potential issues or edge cases

The code context provided includes:
- Primary result: The main code snippet related to the question
- Dependencies: Functions/classes that this code calls
- Dependents: Functions/classes that call this code
- Related code: Other potentially relevant code

Use this context to provide comprehensive answers."""

# System prompt for code search
SEARCH_SYSTEM_PROMPT = """You are CodeMind, a code search and navigation expert. Help developers find and understand code.

When presented with search results:
1. Identify the most relevant match
2. Explain why it matches the query
3. Describe related code that might be useful
4. Suggest how the code relates to other parts of the system

Be precise and reference specific files and line numbers."""

# System prompt for code generation
GENERATION_SYSTEM_PROMPT = """You are CodeMind, an expert code generator. Help developers write better code.

When generating code:
1. Follow the existing code style and patterns
2. Include docstrings and type hints
3. Consider error handling and edge cases
4. Optimize for readability and maintainability
5. Suggest tests if appropriate

Reference the provided code context when generating related functionality."""

# Templates for different query types

QUERY_ANALYSIS_TEMPLATE = """I have a question about this codebase.

Query: {query}

Here is the relevant code context:

{context}

Please answer my question based on this code context. Be thorough but concise."""

QUERY_COMPARISON_TEMPLATE = """Compare these code snippets and explain their differences:

Snippet 1:
{snippet1}

Snippet 2:
{snippet2}

Key differences:
- Purpose
- Performance
- Maintainability
- Use cases"""

QUERY_REFACTORING_TEMPLATE = """Review this code for potential improvements:

{context}

Consider:
- Code style and readability
- Performance optimization opportunities
- Error handling
- Type safety
- Documentation

Suggest specific improvements with explanations."""

QUERY_DEBUGGING_TEMPLATE = """Help me understand why this code might not work as expected:

{context}

Question: {query}

Analyze the code for:
- Logic errors
- Edge cases that might fail
- Type mismatches
- Missing error handling
- Performance issues"""

QUERY_DOCUMENTATION_TEMPLATE = """Generate documentation for this code:

{context}

Provide:
1. High-level summary
2. Function/class descriptions
3. Parameters and return values
4. Usage examples
5. Edge cases and limitations"""

MULTI_RESULT_TEMPLATE = """I found multiple code snippets related to your query.

Query: {query}

Results:

{results}

Summary:
- Most relevant result
- How they relate to each other
- Recommended starting point"""

# Few-shot examples for better responses

EXAMPLE_CODE_QUESTION = """Q: What does this authentication function do?

Code:
```python
def authenticate_user(username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if user and user.verify_password(password):
        return user
    return None
```

A: This function authenticates a user by:
1. Looking up the user in the database by username
2. Verifying the provided password matches the stored hash
3. Returning the User object if authentication succeeds, None otherwise

Key points:
- Uses password hashing verification (more secure than plain comparison)
- Returns early on lookup failure
- Simple and secure pattern"""

EXAMPLE_IMPROVEMENT = """Q: How could this be improved?

Code:
```python
def process_file(filename):
    f = open(filename)
    content = f.read()
    f.close()
    return content
```

A: Several improvements:
1. Use context manager (with statement) for automatic cleanup
2. Add error handling for missing files
3. Add type hints
4. Specify encoding explicitly

Improved:
```python
def process_file(filename: str, encoding: str = 'utf-8') -> str:
    try:
        with open(filename, encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"File not found: {filename}")
```"""

# Build prompts for different scenarios

def get_system_prompt(prompt_type: str = "default") -> str:
    """Get system prompt for given type."""
    prompts = {
        "default": SYSTEM_PROMPT,
        "search": SEARCH_SYSTEM_PROMPT,
        "generation": GENERATION_SYSTEM_PROMPT,
        "analysis": SYSTEM_PROMPT,
    }
    return prompts.get(prompt_type, SYSTEM_PROMPT)


def format_query_prompt(
    query: str,
    context: str,
    prompt_type: str = "analysis"
) -> str:
    """Format a query prompt with context."""
    if prompt_type == "analysis":
        template = QUERY_ANALYSIS_TEMPLATE
    elif prompt_type == "refactoring":
        template = QUERY_REFACTORING_TEMPLATE
    elif prompt_type == "debugging":
        template = QUERY_DEBUGGING_TEMPLATE
    elif prompt_type == "documentation":
        template = QUERY_DOCUMENTATION_TEMPLATE
    else:
        template = QUERY_ANALYSIS_TEMPLATE
    
    return template.format(query=query, context=context)


def format_comparison_prompt(snippet1: str, snippet2: str) -> str:
    """Format a comparison prompt."""
    return QUERY_COMPARISON_TEMPLATE.format(snippet1=snippet1, snippet2=snippet2)


def format_refactoring_prompt(context: str) -> str:
    """Format a refactoring review prompt."""
    return QUERY_REFACTORING_TEMPLATE.format(context=context)


def format_debugging_prompt(context: str, query: str) -> str:
    """Format a debugging prompt."""
    return QUERY_DEBUGGING_TEMPLATE.format(context=context, query=query)


def format_documentation_prompt(context: str) -> str:
    """Format a documentation generation prompt."""
    return QUERY_DOCUMENTATION_TEMPLATE.format(context=context)


def format_multi_result_prompt(query: str, results_text: str) -> str:
    """Format a prompt for multiple search results."""
    return MULTI_RESULT_TEMPLATE.format(query=query, results=results_text)
