#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Chunking Module for RAG Q&A System

Handles French and Arabic insurance documents with heading-aware and paragraph-aware
semantic chunking. Preserves Unicode characters and paragraph structure.

Author: Senior NLP Engineer
Target: Production-ready chunking for nomic-embed-text-v1.5 + FAISS + Llama 3.1
"""

import re
from collections import Counter
from typing import List, Tuple, Set


def _looks_like_heading(line: str) -> bool:
    """
    Detect if a line looks like a heading based on French/Arabic patterns.
    
    Args:
        line: Text line to analyze
        
    Returns:
        True if line appears to be a heading
    """
    line = line.strip()
    if not line:
        return False
    
    # Length-based filtering: very short or very long lines unlikely to be headings
    if len(line) < 3 or len(line) > 200:
        return False
    
    # French heading patterns
    french_patterns = [
        r'^ARTICLE\s+\d+',
        r'^CHAPITRE\s+\d+',
        r'^TITRE\s+[IVX]+',
        r'^DÉFINITIONS?$',
        r'^GARANTIES?$',
        r'^EXCLUSIONS?$',
        r'^FRANCHISES?$',
        r'^CONDITIONS\s+GÉNÉRALES',
        r'^OBLIGATIONS?$',
        r'^DURÉE$',
        r'^RÉSILIATION$',
        r'^SINISTRES?$',
        r'^ANNEXES?$',
        r'^DISPOSITIONS?',
        r'^MODALITÉS?',
        r'^PROCÉDURES?$',
        r'^ASSISTANCE',
        r'^DÉPANNAGE',
        r'^REMORQUAGE',
        r'^PRISE\s+D\'EFFET',
        r'^MISE\s+EN\s+JEU',
        r'^ÉTENDUE\s+DE\s+GARANTIE'
    ]
    
    # Arabic heading patterns
    arabic_patterns = [
        r'مادة\s*\d+',
        r'فصل\s*\d+',
        r'باب\s*\d+',
        r'التعريفات?',
        r'الضمانات?',
        r'التغطيات?',
        r'الاستثناءات?',
        r'الشروط',
        r'الالتزامات?',
        r'المدة',
        r'الفسخ',
        r'الحادث',
        r'التصريح',
        r'الملاحق?',
        r'التأمين\s+ضد\s+الحريق',
        r'المسؤولية\s+المدنية',
        r'المساعدة',
        r'(الجر|السحب)'
    ]
    
    # Check specific patterns
    for pattern in french_patterns + arabic_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    
    # Roman numerals (I. II. III. etc.)
    if re.match(r'^[IVX]+\.\s*', line):
        return True
    
    # Dotted numbering (1.1, 1.1.2, etc.)
    if re.match(r'^\d+(\.\d+)*\.?\s+', line) and len(line) < 100:
        return True
    
    # Lines ending with colon (likely section headers)
    if line.endswith(':') and len(line) < 80:
        return True
    
    # Short ALL-CAPS French lines
    if (len(line) < 60 and 
        line.isupper() and 
        re.search(r'[A-ZÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]', line) and
        not re.search(r'[a-z]', line)):
        return True
    
    # Short Arabic-only lines (likely headings)
    arabic_chars = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', line)
    if (len(line) < 60 and 
        len(arabic_chars) > len(line) * 0.7 and
        not re.search(r'[a-zA-Z]', line)):
        return True
    
    return False


def _detect_repeated_headers(lines: List[str], min_frequency_pct: float = 2.0) -> Set[str]:
    """
    Detect lines that repeat frequently across the document (headers/footers).
    
    Args:
        lines: List of text lines
        min_frequency_pct: Minimum percentage of lines for a line to be considered repeated
        
    Returns:
        Set of repeated header/footer patterns to remove
    """
    if not lines:
        return set()
    
    # Count occurrences of each line
    line_counts = Counter(line.strip() for line in lines if line.strip())
    total_lines = len([line for line in lines if line.strip()])
    
    if total_lines == 0:
        return set()
    
    repeated_lines = set()
    min_occurrences = max(2, int(total_lines * min_frequency_pct / 100))
    
    for line, count in line_counts.items():
        if count >= min_occurrences:
            # Check if it looks like a header/footer pattern
            if (re.search(r'page\s+\d+', line, re.IGNORECASE) or
                re.search(r'\d+/\d+', line) or
                re.search(r'BH\s+Assurance', line, re.IGNORECASE) or
                line.count('-') > 3):
                repeated_lines.add(line)
    
    return repeated_lines


def split_sections(text: str, remove_repeated_headers: bool = True) -> List[Tuple[str, str]]:
    """
    Split text into sections based on detected headings.
    
    Args:
        text: Input text to split
        remove_repeated_headers: Whether to remove repeated header/footer lines
        
    Returns:
        List of (section_title, section_body) tuples
    """
    if not text.strip():
        return [("INTRO", "")]
    
    lines = text.split('\n')
    
    # Remove repeated headers/footers if requested
    if remove_repeated_headers:
        repeated_patterns = _detect_repeated_headers(lines)
        if repeated_patterns:
            lines = [line for line in lines if line.strip() not in repeated_patterns]
    
    sections = []
    current_title = "INTRO"
    current_content = []
    
    for line in lines:
        if _looks_like_heading(line):
            # Save previous section if it has content
            if current_content or not sections:
                section_body = '\n'.join(current_content).strip()
                sections.append((current_title, section_body))
            
            # Start new section
            current_title = line.strip()
            current_content = []
        else:
            current_content.append(line)
    
    # Add final section
    if current_content or not sections:
        section_body = '\n'.join(current_content).strip()
        sections.append((current_title, section_body))
    
    # Filter out empty sections
    sections = [(title, body) for title, body in sections if body.strip()]
    
    # Ensure at least one section exists
    if not sections:
        sections = [("INTRO", text.strip())]
    
    return sections


def paragraph_chunks(text: str, min_len: int = 220, max_len: int = 900, overlap: int = 120) -> List[str]:
    """
    Window paragraphs into target sizes with overlap and strict micro-chunk avoidance.
    
    Args:
        text: Input text to chunk
        min_len: Minimum chunk length in characters
        max_len: Maximum chunk length in characters
        overlap: Overlap between consecutive chunks in characters
        
    Returns:
        List of text chunks with no micro-chunks
    """
    if not text.strip():
        return []
    
    # Split into paragraphs (double newlines)
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return [text.strip()] if text.strip() else []
    
    chunks = []
    current_chunk = ""
    
    for i, paragraph in enumerate(paragraphs):
        # If adding this paragraph would exceed max_len, finalize current chunk
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_len:
            if len(current_chunk) >= min_len:
                chunks.append(current_chunk)
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Current chunk too small, add paragraph anyway
                current_chunk += "\n\n" + paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add final chunk if it exists
    if current_chunk:
        # Check if final chunk is too small and merge with previous if possible
        if len(current_chunk) < min_len and chunks:
            # Merge with last chunk
            last_chunk = chunks.pop()
            merged = last_chunk + "\n\n" + current_chunk
            chunks.append(merged)
        else:
            chunks.append(current_chunk)
    
    # Final pass: merge any remaining micro-chunks
    final_chunks = []
    for chunk in chunks:
        if len(chunk) < min_len and final_chunks:
            # Merge with previous chunk
            last_chunk = final_chunks.pop()
            merged = last_chunk + "\n\n" + chunk
            final_chunks.append(merged)
        else:
            final_chunks.append(chunk)
    
    # Additional pass: merge any remaining micro-chunks at the end
    if len(final_chunks) > 1:
        merged_chunks = []
        for chunk in final_chunks:
            if len(chunk) < min_len and merged_chunks:
                # Merge with previous chunk
                last_chunk = merged_chunks.pop()
                merged = last_chunk + "\n\n" + chunk
                merged_chunks.append(merged)
            else:
                merged_chunks.append(chunk)
        final_chunks = merged_chunks
    
    # Handle case where no chunks meet requirements
    if not final_chunks and text.strip():
        final_chunks = [text.strip()]
    
    return final_chunks


def semantic_chunks(
    text: str,
    min_len: int = 220,
    max_len: int = 900,
    overlap: int = 120,
    remove_repeated_headers: bool = True,
) -> List[Tuple[str, str]]:
    """
    Create semantic chunks using headings and paragraph windowing.
    
    Args:
        text: Input text to chunk
        min_len: Minimum chunk length in characters
        max_len: Maximum chunk length in characters
        overlap: Overlap between consecutive chunks in characters
        remove_repeated_headers: Whether to remove repeated header/footer lines
        
    Returns:
        List of (section_title, chunk_text) tuples
    """
    if not text.strip():
        return [("INTRO", "")]
    
    # Split into sections first
    sections = split_sections(text, remove_repeated_headers)
    
    # Chunk each section
    all_chunks = []
    for section_title, section_body in sections:
        if not section_body.strip():
            continue
            
        # Get paragraph chunks for this section
        section_chunks = paragraph_chunks(section_body, min_len, max_len, overlap)
        
        # Add section title to each chunk
        for chunk_text in section_chunks:
            if chunk_text.strip():
                all_chunks.append((section_title, chunk_text))
    
    # Final micro-chunk merging across sections
    if len(all_chunks) > 1:
        merged_chunks = []
        for section_title, chunk in all_chunks:
            if len(chunk) < min_len and merged_chunks:
                # Merge with previous chunk, keeping the previous section title
                prev_title, prev_chunk = merged_chunks.pop()
                merged_chunk = prev_chunk + "\n\n" + chunk
                merged_chunks.append((prev_title, merged_chunk))
            else:
                merged_chunks.append((section_title, chunk))
        all_chunks = merged_chunks
    
    # Ensure at least one chunk exists
    if not all_chunks:
        all_chunks = [("INTRO", text.strip())]
    
    return all_chunks
