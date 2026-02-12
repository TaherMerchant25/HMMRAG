"""
Entity Resolver - Deduplicate and Merge Entities

This module resolves entity duplicates and merges descriptions.
Reference: LeanRAG's entity resolution approach.

Key Features:
- String normalization (case, punctuation, whitespace)
- Alias detection (abbreviations, common variations)
- Description merging
- Type consolidation
"""

import re
import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResolvedEntity:
    """A resolved entity with merged information."""
    canonical_name: str
    entity_type: str
    descriptions: List[str]
    aliases: Set[str]
    source_ids: Set[str]
    modality: str = 'text'
    
    def __post_init__(self):
        if not isinstance(self.aliases, set):
            self.aliases = set(self.aliases) if self.aliases else set()
        if not isinstance(self.source_ids, set):
            self.source_ids = set(self.source_ids) if self.source_ids else set()
        if not isinstance(self.descriptions, list):
            self.descriptions = list(self.descriptions) if self.descriptions else []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'entity_name': self.canonical_name,
            'type': self.entity_type,
            'description': ' | '.join(self.descriptions) if self.descriptions else f"Entity: {self.canonical_name}",
            'aliases': list(self.aliases),
            'source_ids': list(self.source_ids),
            'modality': self.modality
        }
    
    def merge_description(self, new_desc: str):
        """Add a new description if not duplicate."""
        if new_desc and new_desc not in self.descriptions:
            self.descriptions.append(new_desc)


class EntityResolver:
    """
    Resolve entity duplicates and merge information.
    
    Strategy:
    1. Normalize entity names (lowercase, strip punctuation)
    2. Detect aliases and abbreviations
    3. Merge entities with matching normalized names
    4. Consolidate descriptions and metadata
    """
    
    def __init__(self,
                 case_sensitive: bool = False,
                 merge_threshold: float = 0.9):
        """
        Initialize resolver.
        
        Args:
            case_sensitive: Whether to treat names as case-sensitive
            merge_threshold: Similarity threshold for merging (0-1)
        """
        self.case_sensitive = case_sensitive
        self.merge_threshold = merge_threshold
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize entity name.
        
        Args:
            name: Original entity name
        
        Returns:
            Normalized name
        """
        if not name:
            return ""
        
        # Remove extra whitespace
        normalized = ' '.join(name.split())
        
        # Handle case
        if not self.case_sensitive:
            normalized = normalized.lower()
        
        # Remove common punctuation but keep hyphens and apostrophes
        normalized = re.sub(r'[^\w\s\'-]', '', normalized)
        
        return normalized.strip()
    
    def detect_abbreviation(self, short: str, long: str) -> bool:
        """
        Check if short is an abbreviation of long.
        
        Args:
            short: Potential abbreviation
            long: Full form
        
        Returns:
            True if short is likely an abbreviation of long
        """
        if not short or not long:
            return False
        
        short = short.upper().replace('.', '')
        long_words = long.upper().split()
        
        # Check if short matches first letters of long
        if len(short) == len(long_words):
            return all(s == w[0] for s, w in zip(short, long_words) if w)
        
        # Check if short is in parentheses in long (common pattern)
        return f"({short})" in long.upper() or f"({short.lower()})" in long.lower()
    
    def resolve_triples(self, triples: List[Dict]) -> Tuple[Dict[str, ResolvedEntity], List[Dict]]:
        """
        Resolve entities from triples.
        
        Args:
            triples: List of triple dicts
        
        Returns:
            Tuple of (resolved_entities_dict, resolved_triples)
        """
        # First pass: collect all entity mentions
        entity_mentions = defaultdict(lambda: {
            'names': [],
            'types': [],
            'descriptions': [],
            'source_ids': [],
            'modality': []
        })
        
        for triple in triples:
            # Process head entity
            head = triple.get('head', '')
            if head:
                normalized = self.normalize_name(head)
                entity_mentions[normalized]['names'].append(head)
                entity_mentions[normalized]['types'].append(triple.get('head_type', 'Entity'))
                entity_mentions[normalized]['descriptions'].append(
                    triple.get('head_description', '')
                )
                entity_mentions[normalized]['source_ids'].append(triple.get('source_id', ''))
                entity_mentions[normalized]['modality'].append(
                    triple.get('head_modality', 'text')
                )
            
            # Process tail entity
            tail = triple.get('tail', '')
            if tail:
                normalized = self.normalize_name(tail)
                entity_mentions[normalized]['names'].append(tail)
                entity_mentions[normalized]['types'].append(triple.get('tail_type', 'Entity'))
                entity_mentions[normalized]['descriptions'].append(
                    triple.get('tail_description', '')
                )
                entity_mentions[normalized]['source_ids'].append(triple.get('source_id', ''))
                entity_mentions[normalized]['modality'].append(
                    triple.get('tail_modality', 'text')
                )
        
        # Second pass: create resolved entities
        resolved_entities = {}
        name_to_canonical = {}  # Map all names to canonical name
        
        for normalized_name, mentions in entity_mentions.items():
            # Choose canonical name (most frequent original form)
            name_counts = defaultdict(int)
            for name in mentions['names']:
                name_counts[name] += 1
            canonical_name = max(name_counts.items(), key=lambda x: x[1])[0]
            
            # Choose most specific type
            type_priority = {'Person': 5, 'Organization': 4, 'Theory': 3, 
                           'Concept': 2, 'Entity': 1, 'Unknown': 0}
            entity_type = max(mentions['types'], 
                            key=lambda t: type_priority.get(t, 1))
            
            # Merge descriptions (unique only)
            unique_descriptions = []
            seen = set()
            for desc in mentions['descriptions']:
                if desc and desc not in seen:
                    unique_descriptions.append(desc)
                    seen.add(desc)
            
            # Determine modality (prefer non-text if available)
            modality = 'text'
            if 'image' in mentions['modality']:
                modality = 'image'
            elif 'table' in mentions['modality']:
                modality = 'table'
            elif 'audio' in mentions['modality']:
                modality = 'audio'
            
            # Create resolved entity
            entity = ResolvedEntity(
                canonical_name=canonical_name,
                entity_type=entity_type,
                descriptions=unique_descriptions,
                aliases=set(mentions['names']) - {canonical_name},
                source_ids=set(mentions['source_ids']),
                modality=modality
            )
            
            resolved_entities[canonical_name] = entity
            
            # Map all name variations to canonical name
            for name in mentions['names']:
                name_to_canonical[name] = canonical_name
        
        # Third pass: update triples with canonical names
        resolved_triples = []
        for triple in triples:
            head = triple.get('head', '')
            tail = triple.get('tail', '')
            
            # Map to canonical names
            canonical_head = name_to_canonical.get(head, head)
            canonical_tail = name_to_canonical.get(tail, tail)
            
            # Update triple
            resolved_triple = triple.copy()
            resolved_triple['head'] = canonical_head
            resolved_triple['tail'] = canonical_tail
            
            # Update descriptions and types
            if canonical_head in resolved_entities:
                entity = resolved_entities[canonical_head]
                resolved_triple['head_type'] = entity.entity_type
                resolved_triple['head_description'] = entity.descriptions[0] if entity.descriptions else f"Entity: {canonical_head}"
            
            if canonical_tail in resolved_entities:
                entity = resolved_entities[canonical_tail]
                resolved_triple['tail_type'] = entity.entity_type
                resolved_triple['tail_description'] = entity.descriptions[0] if entity.descriptions else f"Entity: {canonical_tail}"
            
            resolved_triples.append(resolved_triple)
        
        logger.info(f"Resolved {len(entity_mentions)} normalized entities to {len(resolved_entities)} canonical entities")
        logger.info(f"Updated {len(resolved_triples)} triples with canonical names")
        
        return resolved_entities, resolved_triples
    
    def save_entities(self, entities: Dict[str, ResolvedEntity], output_file: str):
        """Save resolved entities to JSONL."""
        with open(output_file, 'w') as f:
            for entity in entities.values():
                f.write(json.dumps(entity.to_dict()) + '\n')
        logger.info(f"Saved {len(entities)} entities to {output_file}")
    
    def save_triples(self, triples: List[Dict], output_file: str):
        """Save resolved triples to JSONL."""
        with open(output_file, 'w') as f:
            for triple in triples:
                f.write(json.dumps(triple) + '\n')
        logger.info(f"Saved {len(triples)} triples to {output_file}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Sample triples with entity duplicates
    triples = [
        {
            'head': 'Albert Einstein',
            'relation': 'developed',
            'tail': 'Theory of Relativity',
            'head_type': 'Person',
            'tail_type': 'Theory',
            'head_description': 'A German physicist',
            'tail_description': 'A fundamental physics theory',
            'source_id': 'doc1'
        },
        {
            'head': 'Einstein',  # Duplicate!
            'relation': 'received',
            'tail': 'Nobel Prize',
            'head_type': 'Person',
            'tail_type': 'Award',
            'head_description': 'Theoretical physicist',
            'tail_description': 'Prestigious award',
            'source_id': 'doc2'
        },
        {
            'head': 'theory of relativity',  # Duplicate with different case!
            'relation': 'is_a',
            'tail': 'Physics Theory',
            'head_type': 'Theory',
            'tail_type': 'Concept',
            'source_id': 'doc3'
        }
    ]
    
    resolver = EntityResolver()
    entities, resolved_triples = resolver.resolve_triples(triples)
    
    print(f"\nResolved Entities ({len(entities)}):\n")
    for name, entity in entities.items():
        print(f"  {name}:")
        print(f"    Type: {entity.entity_type}")
        print(f"    Aliases: {entity.aliases}")
        print(f"    Sources: {len(entity.source_ids)}")
        print()
    
    print(f"\nResolved Triples ({len(resolved_triples)}):\n")
    for triple in resolved_triples:
        print(f"  ({triple['head']}) --[{triple['relation']}]--> ({triple['tail']})")
