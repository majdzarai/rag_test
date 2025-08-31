#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for the multilingual insurance RAG embedding pipeline.

This module centralizes all configuration parameters for document processing,
embedding generation, FAISS indexing, and system limits.
"""

import os
import torch
from pathlib import Path

# ============================================================================
# DATA PATHS
# ============================================================================

# Base data directory containing source documents
DATA_DIR = "../data/txt_pdfs"

# Index storage directory (created if missing)
INDEX_DIR = "../index"

# File names within INDEX_DIR
INDEX_FILE = "index.faiss"          # FAISS index file
META_FILE = "metadata.parquet"      # Chunk metadata table
CHUNKS_FILE = "chunks.parquet"      # Full chunk records

# ============================================================================
# CHUNKING PARAMETERS
# ============================================================================

# Semantic chunking configuration
MIN_LEN = 220                        # Minimum chunk length in characters
MAX_LEN = 900                        # Maximum chunk length in characters
OVERLAP = 120                        # Overlap between chunks in characters

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

# Model configuration
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"  # Hugging Face model identifier
BATCH_SIZE = 64                      # Batch size for embedding generation

# Device selection (auto-detect CUDA availability)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# FAISS INDEX CONFIGURATION
# ============================================================================

# Index type selection
INDEX_TYPE = "hnsw"                # Options: "flatip", "hnsw"

# HNSW-specific parameters (used when INDEX_TYPE="hnsw")
HNSW_M = 64                          # Number of bi-directional links for each node
HNSW_EF_CONSTRUCT = 200              # Size of dynamic candidate list during construction

# MMR (Maximal Marginal Relevance) parameter
MMR_LAMBDA = 0.5                     # Balance between relevance and diversity (0-1)

# ============================================================================
# PROCESSING LIMITS
# ============================================================================

# Text processing limits
MAX_TEXT_CHARS = 8000                # Maximum characters per chunk (truncate with ellipsis)

# Reproducibility
SEED = 42                            # Random seed for deterministic results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_data_dir() -> Path:
    """Get the absolute path to the data directory."""
    return Path(__file__).parent / DATA_DIR

def get_index_dir() -> Path:
    """Get the absolute path to the index directory."""
    return Path(__file__).parent / INDEX_DIR

def get_index_path() -> Path:
    """Get the full path to the FAISS index file."""
    return get_index_dir() / INDEX_FILE

def get_metadata_path() -> Path:
    """Get the full path to the metadata parquet file."""
    return get_index_dir() / META_FILE

def get_chunks_path() -> Path:
    """Get the full path to the chunks parquet file."""
    return get_index_dir() / CHUNKS_FILE

def ensure_index_dir() -> Path:
    """Create index directory if it doesn't exist and return its path."""
    index_dir = get_index_dir()
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir

def validate_config():
    """Validate configuration parameters."""
    if MIN_LEN >= MAX_LEN:
        raise ValueError(f"MIN_LEN ({MIN_LEN}) must be less than MAX_LEN ({MAX_LEN})")
    
    if OVERLAP >= MIN_LEN:
        raise ValueError(f"OVERLAP ({OVERLAP}) must be less than MIN_LEN ({MIN_LEN})")
    
    if not 0 <= MMR_LAMBDA <= 1:
        raise ValueError(f"MMR_LAMBDA ({MMR_LAMBDA}) must be between 0 and 1")
    
    if INDEX_TYPE not in ["flatip", "hnsw"]:
        raise ValueError(f"INDEX_TYPE must be 'flatip' or 'hnsw', got '{INDEX_TYPE}'")
    
    if BATCH_SIZE <= 0:
        raise ValueError(f"BATCH_SIZE ({BATCH_SIZE}) must be positive")
    
    if MAX_TEXT_CHARS <= 0:
        raise ValueError(f"MAX_TEXT_CHARS ({MAX_TEXT_CHARS}) must be positive")

# ============================================================================
# RUNTIME CONFIGURATION
# ============================================================================

# Validate configuration on import
validate_config()

# Set up deterministic behavior
import random
import numpy as np

random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False