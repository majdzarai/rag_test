#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional multilingual embedding retriever for insurance RAG system.

This module provides a production-ready retrieval system with:
- High-performance FAISS search with multiple index types
- Advanced MMR re-ranking for diversity
- Comprehensive error handling and logging
- Performance monitoring and caching
- Clean Python API with type hints
- CLI interface for testing and debugging

Usage:
    # Python API
    retriever = EmbeddingRetriever()
    results = retriever.search("What is covered by insurance?", top_k=5)
    
    # CLI
    python retriever.py --query "insurance coverage" --k 10 --mmr
"""

import argparse
import hashlib
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Local imports
from config_emb import (
    get_index_path, get_chunks_path, get_metadata_path,
    DEVICE, MMR_LAMBDA, MODEL_NAME
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('retriever.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings for cosine similarity via inner product.
    
    Args:
        embeddings: Raw embeddings array
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    return (embeddings / norms).astype(np.float32)

def compute_query_hash(query: str, top_k: int, use_mmr: bool, mmr_lambda: float) -> str:
    """
    Compute hash for query caching.
    
    Args:
        query: Search query
        top_k: Number of results
        use_mmr: Whether MMR is enabled
        mmr_lambda: MMR lambda parameter
        
    Returns:
        Hash string for caching
    """
    cache_key = f"{query}|{top_k}|{use_mmr}|{mmr_lambda:.3f}"
    return hashlib.md5(cache_key.encode('utf-8')).hexdigest()

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.5
) -> List[int]:
    """
    Maximal Marginal Relevance re-ranking for diversity.
    
    Args:
        query_embedding: Query vector (d,)
        candidate_embeddings: Candidate vectors (n, d)
        top_k: Number of results to select
        lambda_mult: Balance between relevance and diversity (0-1)
        
    Returns:
        List of selected candidate indices
    """
    if top_k >= candidate_embeddings.shape[0]:
        return list(range(candidate_embeddings.shape[0]))
    
    # Compute relevance scores (cosine similarity for normalized vectors)
    relevance_scores = candidate_embeddings @ query_embedding
    
    selected_indices = []
    remaining_indices = list(range(candidate_embeddings.shape[0]))
    
    # Select the most relevant item first
    first_idx = int(np.argmax(relevance_scores))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively select items balancing relevance and diversity
    while len(selected_indices) < top_k and remaining_indices:
        selected_embeddings = candidate_embeddings[selected_indices]
        
        # Compute similarity to already selected items
        similarity_to_selected = candidate_embeddings[remaining_indices] @ selected_embeddings.T
        max_similarity = similarity_to_selected.max(axis=1)
        
        # MMR score: balance relevance and diversity
        mmr_scores = (
            lambda_mult * relevance_scores[remaining_indices] -
            (1.0 - lambda_mult) * max_similarity
        )
        
        # Select item with highest MMR score
        best_local_idx = int(np.argmax(mmr_scores))
        best_global_idx = remaining_indices[best_local_idx]
        
        selected_indices.append(best_global_idx)
        remaining_indices.remove(best_global_idx)
    
    return selected_indices

# ============================================================================
# SEARCH RESULT CLASS
# ============================================================================

class SearchResult:
    """
    Represents a single search result with comprehensive metadata.
    """
    
    def __init__(self, chunk_data: Dict, score: float, rank: int):
        """
        Initialize search result.
        
        Args:
            chunk_data: Chunk metadata from DataFrame
            score: Similarity score
            rank: Result rank (1-based)
        """
        self.id = chunk_data.get('id', 'unknown')
        self.text = chunk_data.get('text', '')
        self.section = chunk_data.get('section', 'INTRO')
        self.category = chunk_data.get('category', 'unknown')
        self.filename = chunk_data.get('filename', 'unknown')
        self.file_path = chunk_data.get('file_path', '')
        self.word_count = chunk_data.get('word_count', 0)
        self.char_count = chunk_data.get('char_count', 0)
        self.lang_hint = chunk_data.get('lang_hint', 'unknown')
        self.score = float(score)
        self.rank = int(rank)
        self.last_modified = chunk_data.get('last_modified', '')
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with all result fields
        """
        return {
            'id': self.id,
            'text': self.text,
            'section': self.section,
            'category': self.category,
            'filename': self.filename,
            'file_path': self.file_path,
            'word_count': self.word_count,
            'char_count': self.char_count,
            'lang_hint': self.lang_hint,
            'score': self.score,
            'rank': self.rank,
            'last_modified': self.last_modified
        }
    
    def get_snippet(self, max_chars: int = 200) -> str:
        """
        Get a text snippet for display.
        
        Args:
            max_chars: Maximum characters in snippet
            
        Returns:
            Truncated text snippet
        """
        text = self.text.replace('\n', ' ').strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars-3] + "..."
    
    def __repr__(self) -> str:
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, section='{self.section}')"

# ============================================================================
# MAIN RETRIEVER CLASS
# ============================================================================

class EmbeddingRetriever:
    """
    Professional embedding retriever with comprehensive features.
    """
    
    def __init__(self, cache_size: int = 100, enable_metrics: bool = True):
        """
        Initialize the embedding retriever.
        
        Args:
            cache_size: Maximum number of cached queries
            enable_metrics: Whether to collect performance metrics
        """
        self.cache_size = cache_size
        self.enable_metrics = enable_metrics
        self.query_cache: Dict[str, List[SearchResult]] = {}
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_search_time': 0.0,
            'avg_embedding_time': 0.0,
            'total_search_time': 0.0,
            'total_embedding_time': 0.0
        }
        
        # Load components
        self._load_index()
        self._load_chunks()
        self._load_metadata()
        self._load_encoder()
        
        logger.info("EmbeddingRetriever initialized successfully")
    
    def _load_index(self) -> None:
        """
        Load FAISS index with error handling.
        """
        index_path = get_index_path()
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        try:
            logger.info(f"Loading FAISS index: {index_path}")
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Index loaded: {self.index.ntotal:,} vectors, dimension {self.index.d}")
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")
    
    def _load_chunks(self) -> None:
        """
        Load chunk metadata with validation.
        """
        chunks_path = get_chunks_path()
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        try:
            logger.info(f"Loading chunks: {chunks_path}")
            self.chunks_df = pd.read_parquet(chunks_path)
            
            # Validate chunk count matches index
            if len(self.chunks_df) != self.index.ntotal:
                raise ValueError(
                    f"Chunk count mismatch: {len(self.chunks_df)} chunks vs {self.index.ntotal} vectors"
                )
            
            logger.info(f"Loaded {len(self.chunks_df):,} chunk records")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load chunks: {e}")
    
    def _load_metadata(self) -> None:
        """
        Load build metadata and extract configuration.
        """
        metadata_path = get_metadata_path()
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        try:
            logger.info(f"Loading metadata: {metadata_path}")
            metadata_df = pd.read_parquet(metadata_path)
            
            if len(metadata_df) == 0:
                raise ValueError("Empty metadata file")
            
            # Extract build configuration
            metadata = metadata_df.iloc[0]
            self.model_name = metadata.get('model_name', MODEL_NAME)
            self.index_type = metadata.get('index_type', 'flatip')
            self.embedding_dimension = metadata.get('embedding_dimension', 384)
            self.build_timestamp = metadata.get('build_timestamp', 'unknown')
            
            logger.info(f"Model: {self.model_name}, Index: {self.index_type}, Dim: {self.embedding_dimension}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}")
    
    def _load_encoder(self) -> None:
        """
        Load embedding model with proper configuration.
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.encoder = SentenceTransformer(
                self.model_name,
                device=DEVICE,
                trust_remote_code=True
            )
            
            # Test encoding
            test_embedding = self.encoder.encode(["test"], convert_to_numpy=True)
            if test_embedding.shape[1] != self.embedding_dimension:
                logger.warning(
                    f"Dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {test_embedding.shape[1]}"
                )
            
            logger.info(f"Encoder loaded successfully on {DEVICE}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load encoder: {e}")
    
    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode query text to embedding vector.
        
        Args:
            query: Query text
            
        Returns:
            Normalized embedding vector
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        try:
            # Generate embedding
            embedding = self.encoder.encode(
                [query.strip()],
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False
            )
            
            # Normalize for cosine similarity
            embedding = normalize_embeddings(embedding)
            
            # Update metrics
            if self.enable_metrics:
                encoding_time = time.time() - start_time
                self.metrics['total_embedding_time'] += encoding_time
                self.metrics['avg_embedding_time'] = (
                    self.metrics['total_embedding_time'] / max(1, self.metrics['total_queries'])
                )
            
            return embedding[0]  # Return single vector
            
        except Exception as e:
            raise RuntimeError(f"Query encoding failed: {e}")
    
    def _faiss_search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform FAISS search with error handling.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (scores, indices)
        """
        try:
            # FAISS expects (batch_size, dimension)
            query_batch = query_embedding.reshape(1, -1)
            
            # Perform search
            scores, indices = self.index.search(query_batch, top_k)
            
            return scores[0], indices[0]
            
        except Exception as e:
            raise RuntimeError(f"FAISS search failed: {e}")
    
    def _create_search_results(
        self, 
        scores: np.ndarray, 
        indices: np.ndarray, 
        reranked_order: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """
        Create SearchResult objects from FAISS results.
        
        Args:
            scores: Similarity scores
            indices: Chunk indices
            reranked_order: Optional reranking order for MMR
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        # Apply reranking if provided
        if reranked_order is not None:
            scores = scores[reranked_order]
            indices = indices[reranked_order]
        
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            if idx < 0:  # FAISS returns -1 for invalid results
                continue
            
            try:
                chunk_data = self.chunks_df.iloc[int(idx)].to_dict()
                result = SearchResult(chunk_data, score, rank)
                results.append(result)
            except (IndexError, KeyError) as e:
                logger.warning(f"Invalid chunk index {idx}: {e}")
                continue
        
        return results
    
    def _apply_mmr_reranking(
        self, 
        query_embedding: np.ndarray, 
        results: List[SearchResult], 
        top_k: int, 
        mmr_lambda: float
    ) -> List[SearchResult]:
        """
        Apply MMR re-ranking to search results.
        
        Args:
            query_embedding: Original query vector
            results: Initial search results
            top_k: Final number of results
            mmr_lambda: MMR lambda parameter
            
        Returns:
            Re-ranked search results
        """
        if len(results) <= top_k:
            return results
        
        try:
            # Re-encode result texts for MMR
            result_texts = [result.text for result in results]
            result_embeddings = self.encoder.encode(
                result_texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False
            )
            result_embeddings = normalize_embeddings(result_embeddings)
            
            # Apply MMR
            selected_indices = maximal_marginal_relevance(
                query_embedding, result_embeddings, top_k, mmr_lambda
            )
            
            # Reorder results and update ranks
            reranked_results = []
            for new_rank, idx in enumerate(selected_indices, 1):
                result = results[idx]
                result.rank = new_rank
                reranked_results.append(result)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"MMR re-ranking failed: {e}")
            return results[:top_k]  # Fallback to original ranking
    
    def _update_cache(self, cache_key: str, results: List[SearchResult]) -> None:
        """
        Update query cache with LRU eviction.
        
        Args:
            cache_key: Cache key
            results: Search results to cache
        """
        if self.cache_size <= 0:
            return
        
        # Remove oldest entries if cache is full
        if len(self.query_cache) >= self.cache_size:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = results
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_mmr: bool = False,
        mmr_lambda: float = MMR_LAMBDA,
        mmr_fetch_k: int = 25,
        use_cache: bool = True
    ) -> List[SearchResult]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            use_mmr: Whether to apply MMR re-ranking for diversity
            mmr_lambda: MMR lambda parameter (0=diversity, 1=relevance)
            mmr_fetch_k: Number of candidates to fetch before MMR
            use_cache: Whether to use query caching
            
        Returns:
            List of SearchResult objects ordered by relevance/MMR score
            
        Raises:
            ValueError: If query is empty or parameters are invalid
            RuntimeError: If search fails
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if use_mmr and (mmr_lambda < 0 or mmr_lambda > 1):
            raise ValueError("mmr_lambda must be between 0 and 1")
        
        start_time = time.time()
        
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = compute_query_hash(query, top_k, use_mmr, mmr_lambda)
            if cache_key in self.query_cache:
                self.metrics['cache_hits'] += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return self.query_cache[cache_key][:top_k]
        
        try:
            # Encode query
            query_embedding = self._encode_query(query)
            
            # Determine fetch size
            fetch_k = mmr_fetch_k if use_mmr else top_k
            fetch_k = min(fetch_k, self.index.ntotal)  # Don't exceed index size
            
            # Perform FAISS search
            scores, indices = self._faiss_search(query_embedding, fetch_k)
            
            # Create initial results
            results = self._create_search_results(scores, indices)
            
            # Apply MMR re-ranking if requested
            if use_mmr and len(results) > top_k:
                results = self._apply_mmr_reranking(
                    query_embedding, results, top_k, mmr_lambda
                )
            else:
                results = results[:top_k]
            
            # Update cache
            if use_cache and cache_key:
                self._update_cache(cache_key, results)
            
            # Update metrics
            if self.enable_metrics:
                search_time = time.time() - start_time
                self.metrics['total_queries'] += 1
                self.metrics['total_search_time'] += search_time
                self.metrics['avg_search_time'] = (
                    self.metrics['total_search_time'] / self.metrics['total_queries']
                )
            
            logger.debug(
                f"Search completed: {len(results)} results in {time.time() - start_time:.3f}s"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query[:50]}...': {e}")
            raise RuntimeError(f"Search failed: {e}")
    
    def get_metrics(self) -> Dict:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        metrics = self.metrics.copy()
        metrics.update({
            'cache_hit_rate': (
                self.metrics['cache_hits'] / max(1, self.metrics['total_queries'])
            ) * 100,
            'cache_size': len(self.query_cache),
            'index_size': self.index.ntotal,
            'chunk_count': len(self.chunks_df)
        })
        return metrics
    
    def clear_cache(self) -> None:
        """
        Clear the query cache.
        """
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data dictionary or None if not found
        """
        try:
            chunk_row = self.chunks_df[self.chunks_df['id'] == chunk_id]
            if len(chunk_row) == 0:
                return None
            return chunk_row.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive retriever statistics.
        
        Returns:
            Dictionary with detailed statistics
        """
        try:
            stats = {
                'index_info': {
                    'type': self.index_type,
                    'total_vectors': self.index.ntotal,
                    'dimension': self.embedding_dimension,
                    'model_name': self.model_name
                },
                'chunk_stats': {
                    'total_chunks': len(self.chunks_df),
                    'avg_char_length': self.chunks_df['char_count'].mean(),
                    'categories': self.chunks_df['category'].value_counts().to_dict(),
                    'languages': self.chunks_df['lang_hint'].value_counts().to_dict()
                },
                'performance': self.get_metrics(),
                'build_info': {
                    'timestamp': self.build_timestamp,
                    'device': DEVICE
                }
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to generate stats: {e}")
            return {}

# ============================================================================
# CLI INTERFACE
# ============================================================================

def format_results(results: List[SearchResult], max_snippet_chars: int = 200) -> None:
    """
    Format and print search results to console.
    
    Args:
        results: Search results to display
        max_snippet_chars: Maximum characters in text snippet
    """
    if not results:
        print("\n❌ No results found.")
        return
    
    print(f"\n🔍 Found {len(results)} results:")
    print("=" * 80)
    
    for result in results:
        snippet = result.get_snippet(max_snippet_chars)
        
        print(f"\n[{result.rank}] Score: {result.score:.4f} | {result.filename} | {result.section}")
        print(f"    Language: {result.lang_hint} | Length: {result.char_count} chars")
        print(f"    {snippet}")
        print("-" * 80)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Professional embedding retriever for insurance RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Query parameters
    parser.add_argument('--query', '-q', required=True,
                       help='Search query text')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                       help='Number of results to return')
    
    # MMR parameters
    parser.add_argument('--mmr', action='store_true',
                       help='Enable MMR re-ranking for diversity')
    parser.add_argument('--mmr-lambda', type=float, default=MMR_LAMBDA,
                       help='MMR lambda (0=diversity, 1=relevance)')
    parser.add_argument('--mmr-fetch', type=int, default=25,
                       help='Candidates to fetch before MMR')
    
    # Display options
    parser.add_argument('--snippet-chars', type=int, default=200,
                       help='Maximum characters in result snippets')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable query caching')
    parser.add_argument('--stats', action='store_true',
                       help='Show retriever statistics')
    parser.add_argument('--metrics', action='store_true',
                       help='Show performance metrics')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce logging output')
    
    return parser.parse_args()

def main():
    """
    Main CLI entry point.
    """
    args = parse_arguments()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Initialize retriever
        logger.info("Initializing embedding retriever...")
        retriever = EmbeddingRetriever()
        
        # Show statistics if requested
        if args.stats:
            stats = retriever.get_stats()
            print("\n📊 RETRIEVER STATISTICS")
            print("=" * 50)
            print(f"Index: {stats['index_info']['total_vectors']:,} vectors")
            print(f"Model: {stats['index_info']['model_name']}")
            print(f"Chunks: {stats['chunk_stats']['total_chunks']:,}")
            print(f"Categories: {list(stats['chunk_stats']['categories'].keys())}")
            print(f"Languages: {list(stats['chunk_stats']['languages'].keys())}")
            return
        
        # Perform search
        logger.info(f"Searching for: {args.query}")
        start_time = time.time()
        
        results = retriever.search(
            query=args.query,
            top_k=args.top_k,
            use_mmr=args.mmr,
            mmr_lambda=args.mmr_lambda,
            mmr_fetch_k=args.mmr_fetch,
            use_cache=not args.no_cache
        )
        
        search_time = time.time() - start_time
        
        # Display results
        format_results(results, args.snippet_chars)
        
        # Show metrics if requested
        if args.metrics:
            metrics = retriever.get_metrics()
            print(f"\n⚡ PERFORMANCE METRICS")
            print("=" * 30)
            print(f"Search time: {search_time:.3f}s")
            print(f"Total queries: {metrics['total_queries']}")
            print(f"Cache hit rate: {metrics['cache_hit_rate']:.1f}%")
            print(f"Avg search time: {metrics['avg_search_time']:.3f}s")
        
    except KeyboardInterrupt:
        logger.info("Search interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
