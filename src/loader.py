#!/usr/bin/env python3
"""
Production-Ready Document Loader for RAG Q&A System

This module provides a robust document loader for insurance documents
in French and Arabic languages, preserving Unicode characters and
paragraph structure for optimal RAG performance.

Author: TaaminAI Backend Team
Version: 2.0.0
Date: 2024
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('document_loader.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Production-ready document loader for multilingual insurance documents.
    
    Handles French and Arabic text with full Unicode preservation,
    maintains paragraph structure, and provides comprehensive metadata
    for citation purposes in RAG systems.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the DocumentLoader.
        
        Args:
            base_path (Optional[str]): Base directory path for document scanning.
                                     Defaults to '../data/txt_pdfs' relative to this file.
        """
        if base_path is None:
            # Default to ../data/txt_pdfs relative to this file
            current_dir = Path(__file__).parent
            self.base_path = current_dir.parent / 'data' / 'txt_pdfs'
        else:
            self.base_path = Path(base_path)
        
        self.base_path = self.base_path.resolve()
        logger.info(f"DocumentLoader initialized with base path: {self.base_path}")
        
        # Statistics tracking
        self.stats = {
            'files_found': 0,
            'files_loaded': 0,
            'files_failed': 0,
            'total_chars': 0,
            'total_words': 0,
            'categories': {}
        }
    
    def _validate_path(self) -> bool:
        """
        Validate that the base path exists and is accessible.
        
        Returns:
            bool: True if path is valid, False otherwise.
        """
        if not self.base_path.exists():
            logger.error(f"Base path does not exist: {self.base_path}")
            return False
        
        if not self.base_path.is_dir():
            logger.error(f"Base path is not a directory: {self.base_path}")
            return False
        
        logger.info(f"Base path validated: {self.base_path}")
        return True
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text while preserving Unicode and paragraph structure.
        
        Args:
            text (str): Raw text content.
            
        Returns:
            str: Normalized text with preserved Unicode and structure.
        """
        if not text:
            return ""
        
        # Remove \r characters but keep \n for paragraph breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove control characters except \n and \t
        # Use Unicode categories to preserve all valid characters
        cleaned_chars = []
        for char in text:
            if char in ('\n', '\t'):
                cleaned_chars.append(char)
            elif not unicodedata.category(char).startswith('C'):
                cleaned_chars.append(char)
        
        text = ''.join(cleaned_chars)
        
        # Collapse multiple spaces and tabs, but preserve newlines
        # Replace multiple spaces/tabs with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize multiple consecutive newlines to double newlines (paragraph breaks)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line while preserving structure
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # Remove leading and trailing whitespace from entire text
        text = text.strip()
        
        return text
    
    def _extract_category(self, file_path: Path) -> str:
        """
        Extract category from file path (first directory after txt_pdfs).
        
        Args:
            file_path (Path): Path to the file.
            
        Returns:
            str: Category name.
        """
        try:
            # Get relative path from base_path
            relative_path = file_path.relative_to(self.base_path)
            
            # Get the first part of the path (top-level directory)
            category = relative_path.parts[0] if relative_path.parts else "Unknown"
            
            return category
        except ValueError:
            # File is not under base_path
            logger.warning(f"File not under base path: {file_path}")
            return "Unknown"
    
    def _count_words(self, text: str) -> int:
        """
        Count words in multilingual text (French/Arabic).
        
        Args:
            text (str): Text to count words in.
            
        Returns:
            int: Number of words.
        """
        if not text:
            return 0
        
        # Split on whitespace and filter out empty strings
        words = [word for word in text.split() if word.strip()]
        return len(words)
    
    def _load_single_document(self, file_path: Path) -> Optional[Dict]:
        """
        Load a single document file.
        
        Args:
            file_path (Path): Path to the document file.
            
        Returns:
            Optional[Dict]: Document dictionary or None if failed.
        """
        try:
            # Read file with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                raw_content = file.read()
            
            # Skip empty files
            if not raw_content.strip():
                logger.warning(f"Skipping empty file: {file_path}")
                return None
            
            # Normalize text
            content = self._normalize_text(raw_content)
            
            if not content:
                logger.warning(f"File became empty after normalization: {file_path}")
                return None
            
            # Extract metadata
            category = self._extract_category(file_path)
            char_count = len(content)
            word_count = self._count_words(content)
            
            # Get file modification time
            stat = file_path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            # Create document dictionary
            document = {
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'category': category,
                    'file_path': str(file_path.absolute()),
                    'char_count': char_count,
                    'word_count': word_count,
                    'last_modified': last_modified
                }
            }
            
            # Update statistics
            self.stats['total_chars'] += char_count
            self.stats['total_words'] += word_count
            
            if category not in self.stats['categories']:
                self.stats['categories'][category] = {
                    'files': 0,
                    'chars': 0,
                    'words': 0
                }
            
            self.stats['categories'][category]['files'] += 1
            self.stats['categories'][category]['chars'] += char_count
            self.stats['categories'][category]['words'] += word_count
            
            logger.debug(f"Successfully loaded: {file_path.name} ({char_count} chars, {word_count} words)")
            return document
            
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error in {file_path}: {e}")
            self.stats['files_failed'] += 1
            return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            self.stats['files_failed'] += 1
            return None
    
    def load_documents(self, file_extensions: List[str] = None) -> List[Dict]:
        """
        Load all documents from the base path.
        
        Args:
            file_extensions (List[str], optional): File extensions to load.
                                                 Defaults to ['.txt'].
        
        Returns:
            List[Dict]: List of loaded documents.
        """
        if file_extensions is None:
            file_extensions = ['.txt']
        
        logger.info(f"Starting document loading from: {self.base_path}")
        logger.info(f"Looking for files with extensions: {file_extensions}")
        
        # Validate base path
        if not self._validate_path():
            logger.error("Cannot proceed with invalid base path")
            return []
        
        # Reset statistics
        self.stats = {
            'files_found': 0,
            'files_loaded': 0,
            'files_failed': 0,
            'total_chars': 0,
            'total_words': 0,
            'categories': {}
        }
        
        documents = []
        
        # Find all files with specified extensions
        for extension in file_extensions:
            pattern = f"**/*{extension}"
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    self.stats['files_found'] += 1
                    
                    # Load document
                    document = self._load_single_document(file_path)
                    if document:
                        documents.append(document)
                        self.stats['files_loaded'] += 1
        
        # Log summary
        success_rate = (self.stats['files_loaded'] / self.stats['files_found'] * 100) if self.stats['files_found'] > 0 else 0
        logger.info(f"Document loading completed:")
        logger.info(f"  Files found: {self.stats['files_found']}")
        logger.info(f"  Files loaded: {self.stats['files_loaded']}")
        logger.info(f"  Files failed: {self.stats['files_failed']}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Total characters: {self.stats['total_chars']:,}")
        logger.info(f"  Total words: {self.stats['total_words']:,}")
        
        return documents
    
    def get_document_statistics(self, docs: List[Dict]) -> Dict:
        """
        Get comprehensive statistics about loaded documents.
        
        Args:
            docs (List[Dict]): List of loaded documents.
            
        Returns:
            Dict: Statistics dictionary.
        """
        if not docs:
            return {
                'total_documents': 0,
                'total_characters': 0,
                'total_words': 0,
                'average_chars_per_doc': 0,
                'average_words_per_doc': 0,
                'categories': {},
                'file_extensions': {}
            }
        
        stats = {
            'total_documents': len(docs),
            'total_characters': sum(doc['metadata']['char_count'] for doc in docs),
            'total_words': sum(doc['metadata']['word_count'] for doc in docs),
            'categories': {},
            'file_extensions': {}
        }
        
        # Calculate averages
        stats['average_chars_per_doc'] = stats['total_characters'] / len(docs)
        stats['average_words_per_doc'] = stats['total_words'] / len(docs)
        
        # Category statistics
        for doc in docs:
            category = doc['metadata']['category']
            if category not in stats['categories']:
                stats['categories'][category] = {
                    'count': 0,
                    'total_chars': 0,
                    'total_words': 0
                }
            
            stats['categories'][category]['count'] += 1
            stats['categories'][category]['total_chars'] += doc['metadata']['char_count']
            stats['categories'][category]['total_words'] += doc['metadata']['word_count']
        
        # File extension statistics
        for doc in docs:
            filename = doc['metadata']['filename']
            ext = Path(filename).suffix.lower()
            if ext not in stats['file_extensions']:
                stats['file_extensions'][ext] = 0
            stats['file_extensions'][ext] += 1
        
        return stats
    
    def print_statistics_table(self, docs: List[Dict]):
        """
        Print a formatted statistics table.
        
        Args:
            docs (List[Dict]): List of loaded documents.
        """
        stats = self.get_document_statistics(docs)
        
        print("\n" + "="*80)
        print("📊 DOCUMENT LOADING STATISTICS")
        print("="*80)
        
        print(f"📁 Scan Path: {self.base_path}")
        print(f"📄 Total Documents: {stats['total_documents']:,}")
        print(f"📝 Total Characters: {stats['total_characters']:,}")
        print(f"🔤 Total Words: {stats['total_words']:,}")
        print(f"📊 Avg Chars/Doc: {stats['average_chars_per_doc']:.0f}")
        print(f"📊 Avg Words/Doc: {stats['average_words_per_doc']:.0f}")
        
        if stats['categories']:
            print("\n📂 BY CATEGORY:")
            print("-" * 60)
            print(f"{'Category':<20} {'Files':<8} {'Characters':<12} {'Words':<10}")
            print("-" * 60)
            
            for category, cat_stats in sorted(stats['categories'].items()):
                print(f"{category:<20} {cat_stats['count']:<8} {cat_stats['total_chars']:<12,} {cat_stats['total_words']:<10,}")
        
        print("="*80)


def load_txt_files(folder_path: str) -> List[Dict]:
    """
    Backward-compatibility function for legacy code.
    
    Args:
        folder_path (str): Path to the folder containing txt files.
        
    Returns:
        List[Dict]: List of documents in legacy format.
    """
    logger.info(f"Loading documents using legacy function: {folder_path}")
    
    # Use DocumentLoader
    loader = DocumentLoader(folder_path)
    documents = loader.load_documents(['.txt'])
    
    # Convert to legacy format
    legacy_docs = []
    for doc in documents:
        legacy_doc = {
            'filename': doc['metadata']['filename'],
            'category': doc['metadata']['category'],
            'content': doc['content']
        }
        legacy_docs.append(legacy_doc)
    
    logger.info(f"Converted {len(legacy_docs)} documents to legacy format")
    return legacy_docs


def main():
    """
    Main function for testing and demonstration.
    """
    try:
        print("🚀 TaaminAI Document Loader v2.0")
        print("=" * 50)
        
        # Initialize DocumentLoader with default path
        loader = DocumentLoader()
        
        # Load documents
        print(f"\n📂 Scanning: {loader.base_path}")
        documents = loader.load_documents(['.txt'])
        
        if not documents:
            print("❌ No documents were loaded. Please check:")
            print(f"   • Path exists: {loader.base_path}")
            print(f"   • Contains .txt files")
            print(f"   • Files are readable")
            return 1
        
        # Print statistics
        loader.print_statistics_table(documents)
        
        # Show sample document
        print("\n📄 SAMPLE DOCUMENT:")
        print("-" * 50)
        sample_doc = documents[0]
        print(f"Filename: {sample_doc['metadata']['filename']}")
        print(f"Category: {sample_doc['metadata']['category']}")
        print(f"Characters: {sample_doc['metadata']['char_count']:,}")
        print(f"Words: {sample_doc['metadata']['word_count']:,}")
        print(f"Modified: {sample_doc['metadata']['last_modified']}")
        print("\nContent (first 200 chars):")
        content_preview = sample_doc['content'][:200]
        if len(sample_doc['content']) > 200:
            content_preview += "..."
        print(f'"{content_preview}"')
        
        print("\n✅ Document loading completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
