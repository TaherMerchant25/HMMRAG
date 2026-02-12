"""
Document Chunker - Text Preprocessing for KG Building

This module handles document chunking for knowledge graph construction.
Inspired by LeanRAG's chunking strategy but simplified.

Key Features:
- Sentence-based chunking with sliding window
- Overlap for context preservation
- Metadata tracking (source, position)
- Support for multiple document formats
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    chunk_id: str
    source_id: str
    position: int
    char_start: int
    char_end: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentChunker:
    """
    Chunk documents into manageable pieces for triple extraction.
    
    Strategy:
    - Sentence-based chunking (preserve semantic units)
    - Sliding window with overlap (maintain context)
    - Min/max chunk sizes (avoid too small/large chunks)
    """
    
    def __init__(self,
                 chunk_size: int = 512,
                 overlap: int = 128,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1024):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Sentence splitting pattern (simple but effective)
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def chunk_text(self, text: str, source_id: str = "unknown") -> List[Chunk]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Input text to chunk
            source_id: Identifier for the source document
        
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences
        sentences = self.sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        char_start = 0
        chunk_position = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed max size
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{source_id}_chunk_{chunk_position}"
                
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source_id=source_id,
                    position=chunk_position,
                    char_start=char_start,
                    char_end=char_start + len(chunk_text),
                    metadata={'sentence_count': len(current_chunk)}
                )
                chunks.append(chunk)
                
                # Calculate overlap
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences
                current_length = overlap_length
                char_start = char_start + len(chunk_text) - overlap_length
                chunk_position += 1
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length
            
            # Check if we've reached target chunk size
            if current_length >= self.chunk_size and len(current_chunk) > 1:
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{source_id}_chunk_{chunk_position}"
                
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source_id=source_id,
                    position=chunk_position,
                    char_start=char_start,
                    char_end=char_start + len(chunk_text),
                    metadata={'sentence_count': len(current_chunk)}
                )
                chunks.append(chunk)
                
                # Overlap calculation
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
                char_start = char_start + len(chunk_text) - overlap_length
                chunk_position += 1
        
        # Add final chunk if it meets minimum size
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk_id = f"{source_id}_chunk_{chunk_position}"
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source_id=source_id,
                    position=chunk_position,
                    char_start=char_start,
                    char_end=char_start + len(chunk_text),
                    metadata={'sentence_count': len(current_chunk)}
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {source_id} ({len(text)} chars)")
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of dicts with 'text' and 'id' keys
        
        Returns:
            List of all chunks
        """
        all_chunks = []
        for doc in documents:
            text = doc.get('text', '')
            source_id = doc.get('id', doc.get('source_id', f'doc_{len(all_chunks)}'))
            chunks = self.chunk_text(text, source_id)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)} from {len(documents)} documents")
        return all_chunks


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test chunking
    chunker = DocumentChunker(chunk_size=200, overlap=50)
    
    sample_text = """
    Albert Einstein was a German-born theoretical physicist. He developed the theory 
    of relativity, one of the two pillars of modern physics. Einstein's work is also 
    known for its influence on the philosophy of science. Einstein is best known to 
    the general public for his mass–energy equivalence formula E = mc².
    
    He received the 1921 Nobel Prize in Physics for his services to theoretical 
    physics, and especially for his discovery of the law of the photoelectric effect, 
    a pivotal step in the development of quantum theory. His intellectual achievements 
    and originality have made the word "Einstein" synonymous with "genius".
    """
    
    chunks = chunker.chunk_text(sample_text, "einstein_bio")
    
    print(f"\nCreated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({chunk.chunk_id}):")
        print(f"  Length: {len(chunk.text)} chars")
        print(f"  Text: {chunk.text[:100]}...")
        print()
