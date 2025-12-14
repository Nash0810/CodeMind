"""Setup configuration for CodeMind."""

from setuptools import setup, find_packages

setup(
    name="codemind",
    version="1.0.0",
    description="Intelligent Code Analysis Tool with LLM-Powered Q&A",
    author="Nash",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Parsing & AST
        "tree-sitter==0.25.2",
        "tree-sitter-python==0.25.0",
        # Vector Database & Embeddings
        "chromadb==1.3.5",
        "sentence-transformers==5.1.2",
        "transformers==4.57.3",
        "torch==2.9.1",
        "onnxruntime==1.23.2",
        # Search Engines
        "rank-bm25==0.2.2",
        # HTTP Client
        "requests==2.32.5",
        # CLI & TUI
        "click==8.1.3",
        "rich==14.2.0",
        "textual==6.8.0",
        # Testing
        "pytest==8.0.0",
        "pytest-asyncio==0.23.3",
    ],
    entry_points={
        "console_scripts": [
            "codemind=src.cli.main:cli",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
