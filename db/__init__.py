"""
Database layer for Real Estate RAG system.

Modules:
    config      – DB paths, model config, constants
    normalizer  – Price/area parsing, location splitting
    chunker     – Text chunking + template formatting
    embedder    – Embedding model wrapper
    vectorstore – ChromaDB collections manager
    ingest      – Pipeline: JSON → chunks → embeddings → ChromaDB
"""
