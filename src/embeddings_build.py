#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional embedding pipeline for multilingual (French + Arabic) insurance RAG system.

This module implements a complete pipeline for:
- Document loading and semantic chunking
- Text cleaning and deduplication
- Embedding generation with nomic-ai/nomic-embed-text-v1.5
- FAISS index creation (FlatIP or HNSW)
- Metadata persistence with comprehensive statistics

Usage:
    python embeddings_build.py --rebuild
    python embeddings_build.py --index-type hnsw --sample 5 --rebuild
"""

import argparse
import hashlib
import logging
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Local imports
from config_emb import *
from loader import DocumentLoader
from semantic_chunker import semantic_chunks

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('embeddings_build.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_language(text: str) -> Optional[str]:
    """
    Detect language hint based on Unicode character ranges.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language hint: 'fr', 'ar', 'mix', or None
    """
    # Arabic Unicode range: U+0600-U+06FF
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
    
    # Latin characters (including French accents)
    has_latin = bool(re.search(r'[A-Za-zÀ-ÿ]', text))
    
    if has_arabic and has_latin:
        return "mix"
    elif has_arabic:
        return "ar"
    elif has_latin:
        return "fr"
    else:
        return None

def clean_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """
    Clean and normalize text while preserving Unicode characters.
    
    Args:
        text: Input text to clean
        max_chars: Maximum character limit
        
    Returns:
        Cleaned text
    """
    if not text or not text.strip():
        return ""
    
    # Collapse excessive whitespace while preserving structure
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)     # Collapse spaces/tabs
    text = text.strip()
    
    # Truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars-3] + "..."
    
    return text

def normalize_for_dedup(text: str) -> str:
    """
    Normalize text for deduplication comparison.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text for hashing
    """
    # Convert to lowercase and collapse all whitespace
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return normalized

def compute_text_hash(text: str) -> str:
    """
    Compute SHA-256 hash of normalized text for deduplication.
    
    Args:
        text: Input text
        
    Returns:
        Hexadecimal hash string
    """
    normalized = normalize_for_dedup(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

# ============================================================================
# EMBEDDING PIPELINE
# ============================================================================

class EmbeddingPipeline:
    """
    Professional embedding pipeline with comprehensive error handling and logging.
    """
    
    def __init__(self, config_overrides: Optional[Dict] = None):
        """
        Initialize the embedding pipeline.
        
        Args:
            config_overrides: Optional configuration overrides
        """
        self.config = self._merge_config(config_overrides or {})
        self.model: Optional[SentenceTransformer] = None
        self.stats = {
            'docs_loaded': 0,
            'chunks_created': 0,
            'chunks_after_dedup': 0,
            'chunks_skipped_short': 0,
            'lang_distribution': {'fr': 0, 'ar': 0, 'mix': 0, 'unknown': 0}
        }
    
    def _merge_config(self, overrides: Dict) -> Dict:
        """Merge configuration overrides with defaults."""
        config = {
            'data_dir': DATA_DIR,
            'min_len': MIN_LEN,
            'max_len': MAX_LEN,
            'overlap': OVERLAP,
            'model_name': MODEL_NAME,
            'batch_size': BATCH_SIZE,
            'device': DEVICE,
            'index_type': INDEX_TYPE,
            'hnsw_m': HNSW_M,
            'hnsw_ef_construct': HNSW_EF_CONSTRUCT,
            'max_text_chars': MAX_TEXT_CHARS,
            'seed': SEED
        }
        config.update(overrides)
        return config
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load the embedding model with proper error handling.
        
        Returns:
            Loaded SentenceTransformer model
        """
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.config['model_name']}")
                logger.info(f"Target device: {self.config['device']}")
                
                self.model = SentenceTransformer(
                    self.config['model_name'],
                    device=self.config['device'],
                    trust_remote_code=True
                )
                
                # Test model with a simple encoding
                test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                logger.info(f"Model loaded successfully. Embedding dimension: {test_embedding.shape[1]}")
                
            except Exception as e:
                logger.error(f"Failed to load model {self.config['model_name']}: {e}")
                raise RuntimeError(f"Model loading failed: {e}")
        
        return self.model
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize embeddings for cosine similarity via inner product.
        
        Args:
            embeddings: Raw embeddings array
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Add small epsilon to avoid division by zero
        norms = np.maximum(norms, 1e-12)
        return (embeddings / norms).astype(np.float32)
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts with batching.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Normalized embeddings array
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")
        
        model = self._load_model()
        batch_size = self.config['batch_size']
        
        logger.info(f"Generating embeddings for {len(texts):,} texts (batch_size={batch_size})")
        
        all_embeddings = []
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings without normalization (we'll do it ourselves)
                batch_embeddings = model.encode(
                    batch_texts,
                    normalize_embeddings=False,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batches
            embeddings = np.vstack(all_embeddings)
            
            # Normalize for cosine similarity
            embeddings = self._normalize_embeddings(embeddings)
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            logger.info(f"Embedding statistics: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def _create_chunk_record(self, doc_metadata: Dict, section: str, text: str, 
                           chunk_id: str, file_path: str) -> Dict:
        """
        Create a standardized chunk record with comprehensive metadata.
        
        Args:
            doc_metadata: Document metadata from loader
            section: Section title
            text: Chunk text content
            chunk_id: Unique chunk identifier
            file_path: Absolute file path
            
        Returns:
            Chunk record dictionary
        """
        cleaned_text = clean_text(text, self.config['max_text_chars'])
        
        return {
            'id': chunk_id,
            'text': cleaned_text,
            'section': section,
            'category': doc_metadata.get('category', 'unknown'),
            'filename': doc_metadata.get('filename', 'unknown'),
            'file_path': str(file_path),
            'word_count': len(cleaned_text.split()),
            'char_count': len(cleaned_text),
            'last_modified': datetime.now().isoformat(),
            'lang_hint': detect_language(cleaned_text),
            'start_idx': None,  # Reserved for future use
            'end_idx': None     # Reserved for future use
        }
    
    def _process_documents(self, sample_limit: Optional[int] = None, 
                          enable_dedup: bool = True) -> List[Dict]:
        """
        Load and process documents into chunks with metadata.
        
        Args:
            sample_limit: Limit processing to first N files (for debugging)
            enable_dedup: Whether to enable deduplication
            
        Returns:
            List of chunk records
        """
        logger.info("Starting document processing")
        
        # Load documents
        data_dir = get_data_dir()
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        loader = DocumentLoader(str(data_dir))
        docs = loader.load_documents(['.txt'])
        
        if not docs:
            raise RuntimeError(f"No documents found in {data_dir}")
        
        self.stats['docs_loaded'] = len(docs)
        logger.info(f"Loaded {len(docs):,} documents")
        
        # Apply sample limit if specified
        if sample_limit and sample_limit < len(docs):
            docs = docs[:sample_limit]
            logger.info(f"Limited to first {sample_limit} documents for debugging")
        
        # Process documents into chunks
        chunk_records = []
        seen_hashes: Set[str] = set()
        
        for doc in tqdm(docs, desc="Processing documents"):
            try:
                # Generate semantic chunks
                chunks = semantic_chunks(
                    doc['content'],
                    min_len=self.config['min_len'],
                    max_len=self.config['max_len'],
                    overlap=self.config['overlap'],
                    remove_repeated_headers=True
                )
                
                # Process each chunk
                for section, text in chunks:
                    # Skip very short chunks
                    if len(text.strip()) < 50:
                        self.stats['chunks_skipped_short'] += 1
                        logger.warning(f"Skipping short chunk ({len(text)} chars): {text[:50]}...")
                        continue
                    
                    # Create chunk record
                    chunk_id = str(uuid.uuid4())
                    file_path = doc['metadata'].get('file_path', 'unknown')
                    
                    record = self._create_chunk_record(
                        doc['metadata'], section, text, chunk_id, file_path
                    )
                    
                    # Deduplication check
                    if enable_dedup:
                        text_hash = compute_text_hash(record['text'])
                        if text_hash in seen_hashes:
                            logger.debug(f"Skipping duplicate chunk: {record['id']}")
                            continue
                        seen_hashes.add(text_hash)
                    
                    # Update language statistics
                    lang = record['lang_hint'] or 'unknown'
                    self.stats['lang_distribution'][lang] += 1
                    
                    chunk_records.append(record)
                    self.stats['chunks_created'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process document {doc['metadata'].get('filename', 'unknown')}: {e}")
                continue
        
        self.stats['chunks_after_dedup'] = len(chunk_records)
        
        if not chunk_records:
            raise RuntimeError("No valid chunks were created")
        
        logger.info(f"Created {len(chunk_records):,} chunks after processing")
        return chunk_records
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create FAISS index based on configuration.
        
        Args:
            embeddings: Normalized embeddings array
            
        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        index_type = self.config['index_type']
        
        logger.info(f"Creating FAISS index: type={index_type}, dimension={dimension}")
        
        if index_type == "flatip":
            # Flat index with inner product (cosine similarity on normalized vectors)
            index = faiss.IndexFlatIP(dimension)
            
        elif index_type == "hnsw":
            # HNSW index for faster approximate search
            index = faiss.IndexHNSWFlat(dimension, self.config['hnsw_m'])
            index.hnsw.efConstruction = self.config['hnsw_ef_construct']
            
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add embeddings to index
        logger.info(f"Adding {len(embeddings):,} vectors to index")
        index.add(embeddings)
        
        logger.info(f"Index created successfully. Total vectors: {index.ntotal:,}")
        return index
    
    def _save_results(self, index: faiss.Index, chunk_records: List[Dict], 
                     embeddings: np.ndarray) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index: FAISS index
            chunk_records: List of chunk records
            embeddings: Embeddings array for statistics
        """
        # Ensure index directory exists
        index_dir = ensure_index_dir()
        
        # Save FAISS index
        index_path = get_index_path()
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, str(index_path))
        
        # Save chunk records
        chunks_path = get_chunks_path()
        logger.info(f"Saving chunk records to {chunks_path}")
        chunks_df = pd.DataFrame(chunk_records)
        chunks_df.to_parquet(chunks_path, index=False)
        
        # Create and save metadata summary
        metadata_path = get_metadata_path()
        logger.info(f"Saving metadata summary to {metadata_path}")
        
        # Calculate summary statistics
        summary_stats = {
            'build_timestamp': datetime.now().isoformat(),
            'total_documents': self.stats['docs_loaded'],
            'total_chunks': len(chunk_records),
            'chunks_after_dedup': self.stats['chunks_after_dedup'],
            'chunks_skipped_short': self.stats['chunks_skipped_short'],
            'avg_chunk_length': chunks_df['char_count'].mean(),
            'min_chunk_length': chunks_df['char_count'].min(),
            'max_chunk_length': chunks_df['char_count'].max(),
            'embedding_dimension': embeddings.shape[1],
            'embedding_norm_mean': float(np.linalg.norm(embeddings, axis=1).mean()),
            'embedding_norm_std': float(np.linalg.norm(embeddings, axis=1).std()),
            'index_type': self.config['index_type'],
            'model_name': self.config['model_name'],
            'lang_distribution': self.stats['lang_distribution'],
            'category_distribution': chunks_df['category'].value_counts().to_dict(),
            'chunks_with_headings_pct': (chunks_df['section'] != 'INTRO').mean() * 100
        }
        
        # Save as single-row DataFrame
        metadata_df = pd.DataFrame([summary_stats])
        metadata_df.to_parquet(metadata_path, index=False)
        
        logger.info("All files saved successfully")
    
    def build(self, sample_limit: Optional[int] = None, enable_dedup: bool = True, 
             rebuild: bool = False) -> None:
        """
        Execute the complete embedding pipeline.
        
        Args:
            sample_limit: Limit to first N documents (for debugging)
            enable_dedup: Enable text deduplication
            rebuild: Force rebuild even if files exist
        """
        try:
            # Check if index already exists
            if not rebuild and get_index_path().exists():
                logger.error("Index files already exist. Use --rebuild to overwrite.")
                sys.exit(1)
            
            logger.info("Starting embedding pipeline build")
            logger.info(f"Configuration: {self.config}")
            
            # Process documents
            chunk_records = self._process_documents(sample_limit, enable_dedup)
            
            # Extract texts for embedding
            texts = [record['text'] for record in chunk_records]
            
            # Generate embeddings
            embeddings = self._embed_texts(texts)
            
            # Create FAISS index
            index = self._create_faiss_index(embeddings)
            
            # Save results
            self._save_results(index, chunk_records, embeddings)
            
            # Print final report
            self._print_build_report(chunk_records, embeddings)
            
            logger.info("Embedding pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _print_build_report(self, chunk_records: List[Dict], embeddings: np.ndarray) -> None:
        """
        Print a comprehensive build report to console.
        
        Args:
            chunk_records: List of chunk records
            embeddings: Embeddings array
        """
        chunks_df = pd.DataFrame(chunk_records)
        
        print("\n" + "=" * 80)
        print("EMBEDDING PIPELINE BUILD REPORT")
        print("=" * 80)
        
        print(f"\n📊 DOCUMENT STATISTICS:")
        print(f"  Documents loaded: {self.stats['docs_loaded']:,}")
        print(f"  Total chunks created: {self.stats['chunks_created']:,}")
        print(f"  Chunks after deduplication: {self.stats['chunks_after_dedup']:,}")
        print(f"  Short chunks skipped: {self.stats['chunks_skipped_short']:,}")
        
        print(f"\n📏 CHUNK LENGTH STATISTICS:")
        print(f"  Average length: {chunks_df['char_count'].mean():.1f} chars")
        print(f"  Min length: {chunks_df['char_count'].min():,} chars")
        print(f"  Max length: {chunks_df['char_count'].max():,} chars")
        print(f"  Chunks with headings: {(chunks_df['section'] != 'INTRO').mean() * 100:.1f}%")
        
        print(f"\n🌍 LANGUAGE DISTRIBUTION:")
        for lang, count in self.stats['lang_distribution'].items():
            if count > 0:
                pct = (count / len(chunk_records)) * 100
                print(f"  {lang.upper()}: {count:,} chunks ({pct:.1f}%)")
        
        print(f"\n📂 CATEGORY DISTRIBUTION:")
        for category, count in chunks_df['category'].value_counts().items():
            print(f"  {category}: {count:,} chunks")
        
        print(f"\n🧠 EMBEDDING STATISTICS:")
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  Dimension: {embeddings.shape[1]}")
        print(f"  Norm mean±std: {norms.mean():.4f}±{norms.std():.4f}")
        print(f"  Index type: {self.config['index_type']}")
        print(f"  Model: {self.config['model_name']}")
        
        print(f"\n💾 FILES CREATED:")
        print(f"  FAISS index: {get_index_path()}")
        print(f"  Chunk records: {get_chunks_path()}")
        print(f"  Metadata: {get_metadata_path()}")
        
        print("\n✅ BUILD COMPLETED SUCCESSFULLY")
        print("=" * 80)

# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Build embeddings for multilingual insurance RAG system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and processing
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory containing source documents')
    parser.add_argument('--sample', type=int, metavar='N',
                       help='Process only first N documents (for debugging)')
    parser.add_argument('--no-dedupe', action='store_true',
                       help='Disable text deduplication')
    
    # Chunking parameters
    parser.add_argument('--min-len', type=int, default=MIN_LEN,
                       help='Minimum chunk length in characters')
    parser.add_argument('--max-len', type=int, default=MAX_LEN,
                       help='Maximum chunk length in characters')
    parser.add_argument('--overlap', type=int, default=OVERLAP,
                       help='Overlap between chunks in characters')
    
    # Embedding parameters
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                       help='Embedding model name')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for embedding generation')
    parser.add_argument('--device', type=str, default=DEVICE,
                       help='Device for model inference (cpu/cuda)')
    
    # Index parameters
    parser.add_argument('--index-type', choices=['flatip', 'hnsw'], default=INDEX_TYPE,
                       help='FAISS index type')
    parser.add_argument('--mmr-lambda', type=float, default=MMR_LAMBDA,
                       help='MMR lambda parameter for diversity')
    
    # Control flags
    parser.add_argument('--rebuild', action='store_true',
                       help='Force rebuild even if index exists')
    parser.add_argument('--silent', action='store_true',
                       help='Reduce logging output')
    
    return parser.parse_args()

def main():
    """
    Main entry point for the embedding build pipeline.
    """
    args = parse_arguments()
    
    # Adjust logging level
    if args.silent:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Prepare configuration overrides
    config_overrides = {
        'data_dir': args.data_dir,
        'min_len': args.min_len,
        'max_len': args.max_len,
        'overlap': args.overlap,
        'model_name': args.model,
        'batch_size': args.batch_size,
        'device': args.device,
        'index_type': args.index_type
    }
    
    try:
        # Initialize and run pipeline
        pipeline = EmbeddingPipeline(config_overrides)
        pipeline.build(
            sample_limit=args.sample,
            enable_dedup=not args.no_dedupe,
            rebuild=args.rebuild
        )
        
    except KeyboardInterrupt:
        logger.info("Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
