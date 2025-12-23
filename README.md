# Health Insurance Enrollment Assistant (LangChain Version)

A RAG (Retrieval-Augmented Generation) system for answering questions about health insurance using LangChain.

## Overview

This project implements the same Health Insurance RAG system but uses LangChain framework instead of building from scratch.

**Purpose:** Compare LangChain's abstractions with the from-scratch implementation to understand:
- What frameworks provide
- Trade-offs between control and convenience
- When to use frameworks vs. building custom solutions

## Original Project

This is a rebuild of the from-scratch version at:
- `/home/bhargav/health-enrollment-assistant-dev`

## Tech Stack

- **Framework:** LangChain
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace (sentence-transformers)
- **LLM:** Google Gemini 2.0 Flash
- **UI:** Gradio

## Project Status

Status: Initial setup

## Setup

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

## Usage

Coming soon...

## Comparison with From-Scratch Version

**From Scratch:**
- Manual implementation of all components
- Full control over every detail
- ~15 modules built from ground up
- Deep understanding of RAG internals

**LangChain Version (This Repo):**
- Framework abstractions
- Simpler, less code
- Standard patterns
- Faster development

## Author

Bhargav Sudhir Shivaram

## License

MIT
