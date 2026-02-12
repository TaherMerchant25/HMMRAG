"""
Triple Extractor - Extract Knowledge Graph Triples from Text

This module extracts (head, relation, tail) triples from text chunks.
Reference: LeanRAG's NER+RE approach, but simplified without heavy LLM dependency.

Extraction Methods:
1. Pattern-based extraction (for common patterns)
2. Simple dependency parsing (if available)
3. LLM-based extraction (optional, for complex cases)

Note: For production use, integrate with DeepSeek/GLM like original LeanRAG.
This version provides a foundation that can be extended.
"""

import re
import json
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    """A knowledge graph triple."""
    head: str
    relation: str
    tail: str
    head_type: str = "Entity"
    tail_type: str = "Entity"
    head_description: str = ""
    tail_description: str = ""
    source_id: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'head': self.head,
            'relation': self.relation,
            'tail': self.tail,
            'head_type': self.head_type,
            'tail_type': self.tail_type,
            'head_description': self.head_description or f"Entity: {self.head}",
            'tail_description': self.tail_description or f"Entity: {self.tail}",
            'source_id': self.source_id,
            'confidence': self.confidence,
            'head_modality': 'text',
            'tail_modality': 'text'
        }


class TripleExtractor:
    """
    Extract knowledge graph triples from text.
    
    This is a simplified version. For production:
    - Use LLM-based extraction (DeepSeek, GLM, GPT-4)
    - Use specialized NER+RE models
    - Use dependency parsing (spaCy, etc.)
    """
    
    def __init__(self, use_patterns: bool = True):
        """
        Initialize extractor.
        
        Args:
            use_patterns: Whether to use pattern-based extraction
        """
        self.use_patterns = use_patterns
        
        # Common relation patterns
        self.relation_patterns = [
            # IS-A relations
            (r'(\w+(?:\s+\w+)*)\s+is\s+a[n]?\s+(\w+(?:\s+\w+)*)', 'is_a'),
            (r'(\w+(?:\s+\w+)*)\s+is\s+the\s+(\w+(?:\s+\w+)*)', 'is_a'),
            (r'(\w+(?:\s+\w+)*)\s+are\s+(\w+(?:\s+\w+)*)', 'is_a'),
            
            # ACTION relations
            (r'(\w+(?:\s+\w+)*)\s+developed\s+(\w+(?:\s+\w+)*)', 'developed'),
            (r'(\w+(?:\s+\w+)*)\s+created\s+(\w+(?:\s+\w+)*)', 'created'),
            (r'(\w+(?:\s+\w+)*)\s+invented\s+(\w+(?:\s+\w+)*)', 'invented'),
            (r'(\w+(?:\s+\w+)*)\s+discovered\s+(\w+(?:\s+\w+)*)', 'discovered'),
            (r'(\w+(?:\s+\w+)*)\s+wrote\s+(\w+(?:\s+\w+)*)', 'wrote'),
            (r'(\w+(?:\s+\w+)*)\s+published\s+(\w+(?:\s+\w+)*)', 'published'),
            
            # RELATION relations
            (r'(\w+(?:\s+\w+)*)\s+works?\s+at\s+(\w+(?:\s+\w+)*)', 'works_at'),
            (r'(\w+(?:\s+\w+)*)\s+lives?\s+in\s+(\w+(?:\s+\w+)*)', 'lives_in'),
            (r'(\w+(?:\s+\w+)*)\s+part\s+of\s+(\w+(?:\s+\w+)*)', 'part_of'),
            (r'(\w+(?:\s+\w+)*)\s+belongs?\s+to\s+(\w+(?:\s+\w+)*)', 'belongs_to'),
            (r'(\w+(?:\s+\w+)*)\s+located\s+in\s+(\w+(?:\s+\w+)*)', 'located_in'),
            
            # INFLUENCE relations
            (r'(\w+(?:\s+\w+)*)\s+influenced?\s+(\w+(?:\s+\w+)*)', 'influenced'),
            (r'(\w+(?:\s+\w+)*)\s+based\s+on\s+(\w+(?:\s+\w+)*)', 'based_on'),
            (r'(\w+(?:\s+\w+)*)\s+derived\s+from\s+(\w+(?:\s+\w+)*)', 'derived_from'),
            
            # PROPERTY relations  
            (r'(\w+(?:\s+\w+)*)\s+has\s+(\w+(?:\s+\w+)*)', 'has'),
            (r'(\w+(?:\s+\w+)*)\s+contains?\s+(\w+(?:\s+\w+)*)', 'contains'),
            (r'(\w+(?:\s+\w+)*)\s+includes?\s+(\w+(?:\s+\w+)*)', 'includes'),
        ]
        
        # Entity type patterns (simple heuristics)
        self.type_patterns = {
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b': 'Person',  # Capitalized words
            r'\b\d{4}\b': 'Year',
            r'\b[A-Z]{2,}\b': 'Acronym',
        }
    
    def extract_from_text(self, text: str, source_id: str = "") -> List[Triple]:
        """
        Extract triples from text.
        
        Args:
            text: Input text
            source_id: Source identifier
        
        Returns:
            List of extracted triples
        """
        triples = []
        
        if self.use_patterns:
            triples.extend(self._extract_with_patterns(text, source_id))
        
        # Deduplicate
        unique_triples = {}
        for triple in triples:
            key = (triple.head.lower(), triple.relation, triple.tail.lower())
            if key not in unique_triples:
                unique_triples[key] = triple
        
        return list(unique_triples.values())
    
    def _extract_with_patterns(self, text: str, source_id: str) -> List[Triple]:
        """Extract triples using regex patterns."""
        triples = []
        
        for pattern, relation in self.relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                head = match.group(1).strip()
                tail = match.group(2).strip()
                
                # Skip if too short or too long
                if len(head) < 2 or len(tail) < 2:
                    continue
                if len(head) > 50 or len(tail) > 50:
                    continue
                
                # Infer types (basic heuristics)
                head_type = self._infer_type(head)
                tail_type = self._infer_type(tail)
                
                triple = Triple(
                    head=head,
                    relation=relation,
                    tail=tail,
                    head_type=head_type,
                    tail_type=tail_type,
                    source_id=source_id,
                    confidence=0.8  # Pattern-based has lower confidence
                )
                triples.append(triple)
        
        return triples
    
    def _infer_type(self, entity: str) -> str:
        """Infer entity type from text (basic heuristics)."""
        # Check if it's a year
        if re.match(r'^\d{4}$', entity):
            return 'Year'
        
        # Check if it's capitalized (likely a proper noun)
        if entity[0].isupper():
            # Common patterns
            if any(word in entity.lower() for word in ['theory', 'law', 'principle', 'equation']):
                return 'Theory'
            elif any(word in entity.lower() for word in ['university', 'institute', 'lab']):
                return 'Organization'
            elif re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', entity):
                return 'Person'  # Multi-word capitalized likely a person
            else:
                return 'Concept'
        
        return 'Entity'
    
    def extract_batch(self, chunks: List[Dict]) -> List[Triple]:
        """
        Extract triples from multiple chunks.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'chunk_id'
        
        Returns:
            List of all extracted triples
        """
        all_triples = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            chunk_id = chunk.get('chunk_id', chunk.get('id', ''))
            
            triples = self.extract_from_text(text, chunk_id)
            all_triples.extend(triples)
        
        logger.info(f"Extracted {len(all_triples)} triples from {len(chunks)} chunks")
        return all_triples
    
    def save_triples(self, triples: List[Triple], output_file: str):
        """Save triples to JSONL file."""
        with open(output_file, 'w') as f:
            for triple in triples:
                f.write(json.dumps(triple.to_dict()) + '\n')
        logger.info(f"Saved {len(triples)} triples to {output_file}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = TripleExtractor()
    
    sample_text = """
    Albert Einstein was a German-born theoretical physicist. Einstein developed the 
    theory of relativity. He received the 1921 Nobel Prize in Physics. The theory of 
    relativity is a fundamental theory in physics. Einstein's work influenced modern 
    quantum mechanics. Quantum mechanics is a branch of physics.
    """
    
    triples = extractor.extract_from_text(sample_text, "einstein_sample")
    
    print(f"\nExtracted {len(triples)} triples:\n")
    for triple in triples:
        print(f"  ({triple.head}) --[{triple.relation}]--> ({triple.tail})")
        print(f"    Types: {triple.head_type}, {triple.tail_type}")
        print()
