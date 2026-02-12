"""
Knowledge Graph Builder - End-to-End Pipeline

This module provides a complete pipeline from raw text to knowledge graph:
1. Chunking → 2. Triple Extraction → 3. Entity Resolution → 4. Taxonomy Building

Reference: Inspired by LeanRAG but with LCA-optimized hierarchy.
"""

import os
import json
import logging
from typing import List, Dict
from pathlib import Path

from chunker import DocumentChunker
from triple_extractor import TripleExtractor
from entity_resolver import EntityResolver
from taxonomy_builder import TaxonomyBuilder
from sparse_table import EulerTourLCA
from wu_palmer import WuPalmerSimilarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    End-to-end knowledge graph builder.
    
    Pipeline:
    1. Document Chunking
    2. Triple Extraction  
    3. Entity Resolution
    4. Taxonomy Construction
    5. LCA Structure Building
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize KG builder.
        
        Args:
            config: Configuration dict with chunking, extraction params
        """
        self.config = config or self._default_config()
        
        # Initialize components
        self.chunker = DocumentChunker(
            chunk_size=self.config['chunking']['chunk_size'],
            overlap=self.config['chunking']['overlap']
        )
        
        self.extractor = TripleExtractor(
            use_patterns=self.config['extraction']['use_patterns']
        )
        
        self.resolver = EntityResolver(
            case_sensitive=self.config['resolution']['case_sensitive']
        )
        
        self.taxonomy_builder = TaxonomyBuilder()
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'chunking': {
                'chunk_size': 512,
                'overlap': 128,
                'min_chunk_size': 100
            },
            'extraction': {
                'use_patterns': True,
                'use_llm': False  # Set to True if using LLM API
            },
            'resolution': {
                'case_sensitive': False,
                'merge_threshold': 0.9
            },
            'taxonomy': {
                'virtual_root': 'ROOT'
            }
        }
    
    def build_from_text(self, text: str, source_id: str = "document") -> Dict:
        """
        Build KG from single text document.
        
        Args:
            text: Input text
            source_id: Document identifier
        
        Returns:
            Dict with taxonomy, entities, triples
        """
        logger.info(f"Building KG from {source_id}...")
        
        # Step 1: Chunk
        logger.info("Step 1: Chunking document...")
        chunks = self.chunker.chunk_text(text, source_id)
        logger.info(f"  Created {len(chunks)} chunks")
        
        # Step 2: Extract triples
        logger.info("Step 2: Extracting triples...")
        chunk_dicts = [{'text': c.text, 'chunk_id': c.chunk_id} for c in chunks]
        raw_triples = self.extractor.extract_batch(chunk_dicts)
        logger.info(f"  Extracted {len(raw_triples)} raw triples")
        
        # Step 3: Resolve entities
        logger.info("Step 3: Resolving entities...")
        triple_dicts = [t.to_dict() for t in raw_triples]
        entities, resolved_triples = self.resolver.resolve_triples(triple_dicts)
        logger.info(f"  Resolved to {len(entities)} unique entities")
        logger.info(f"  Updated {len(resolved_triples)} triples")
        
        # Step 4: Build taxonomy
        logger.info("Step 4: Building taxonomy...")
        num_nodes = self.taxonomy_builder.build_from_triples(resolved_triples)
        logger.info(f"  Built taxonomy with {num_nodes} nodes")
        
        # Step 5: Build LCA structure
        logger.info("Step 5: Building LCA structure...")
        lca_solver = EulerTourLCA(self.taxonomy_builder)
        lca_solver.build()
        logger.info(f"  LCA structure ready (O(1) queries)")
        
        # Step 6: Initialize Wu-Palmer similarity
        wp_similarity = WuPalmerSimilarity(lca_solver)
        logger.info(f"  Wu-Palmer similarity ready")
        
        return {
            'taxonomy': self.taxonomy_builder,
            'lca_solver': lca_solver,
            'wp_similarity': wp_similarity,
            'entities': entities,
            'triples': resolved_triples,
            'num_chunks': len(chunks),
            'num_triples': len(resolved_triples),
            'num_entities': len(entities),
            'num_nodes': num_nodes
        }
    
    def build_from_documents(self, documents: List[Dict]) -> Dict:
        """
        Build KG from multiple documents.
        
        Args:
            documents: List of dicts with 'text' and 'id' keys
        
        Returns:
            Dict with taxonomy, entities, triples
        """
        logger.info(f"Building KG from {len(documents)} documents...")
        
        # Step 1: Chunk all documents
        logger.info("Step 1: Chunking documents...")
        all_chunks = self.chunker.chunk_documents(documents)
        logger.info(f"  Created {len(all_chunks)} total chunks")
        
        # Step 2: Extract triples
        logger.info("Step 2: Extracting triples...")
        chunk_dicts = [{'text': c.text, 'chunk_id': c.chunk_id} for c in all_chunks]
        raw_triples = self.extractor.extract_batch(chunk_dicts)
        logger.info(f"  Extracted {len(raw_triples)} raw triples")
        
        # Step 3: Resolve entities
        logger.info("Step 3: Resolving entities...")
        triple_dicts = [t.to_dict() for t in raw_triples]
        entities, resolved_triples = self.resolver.resolve_triples(triple_dicts)
        logger.info(f"  Resolved to {len(entities)} unique entities")
        
        # Step 4: Build taxonomy
        logger.info("Step 4: Building taxonomy...")
        num_nodes = self.taxonomy_builder.build_from_triples(resolved_triples)
        logger.info(f"  Built taxonomy with {num_nodes} nodes")
        
        # Step 5: Build LCA structure
        logger.info("Step 5: Building LCA structure...")
        lca_solver = EulerTourLCA(self.taxonomy_builder)
        lca_solver.build()
        logger.info(f"  LCA structure ready")
        
        # Step 6: Initialize Wu-Palmer similarity
        wp_similarity = WuPalmerSimilarity(lca_solver)
        
        return {
            'taxonomy': self.taxonomy_builder,
            'lca_solver': lca_solver,
            'wp_similarity': wp_similarity,
            'entities': entities,
            'triples': resolved_triples,
            'num_chunks': len(all_chunks),
            'num_documents': len(documents),
            'num_triples': len(resolved_triples),
            'num_entities': len(entities),
            'num_nodes': num_nodes
        }
    
    def save(self, result: Dict, output_dir: str):
        """
        Save KG to disk.
        
        Args:
            result: Result dict from build_from_text/documents
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save entities
        entity_file = os.path.join(output_dir, 'entities.jsonl')
        self.resolver.save_entities(result['entities'], entity_file)
        
        # Save triples
        triple_file = os.path.join(output_dir, 'triples.jsonl')
        self.resolver.save_triples(result['triples'], triple_file)
        
        # Save taxonomy
        taxonomy_file = os.path.join(output_dir, 'taxonomy.json')
        result['taxonomy'].save(taxonomy_file)
        
        # Save statistics
        stats = {
            'num_documents': result.get('num_documents', 1),
            'num_chunks': result['num_chunks'],
            'num_entities': result['num_entities'],
            'num_triples': result['num_triples'],
            'num_taxonomy_nodes': result['num_nodes']
        }
        
        stats_file = os.path.join(output_dir, 'statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved KG to {output_dir}")
        logger.info(f"  - {stats['num_entities']} entities")
        logger.info(f"  - {stats['num_triples']} triples")
        logger.info(f"  - {stats['num_taxonomy_nodes']} taxonomy nodes")


# Example usage and demo
def demo():
    """Run a simple demo."""
    print("\n" + "="*60)
    print("Knowledge Graph Builder Demo")
    print("="*60 + "\n")
    
    # Sample documents
    documents = [
        {
            'id': 'einstein_bio',
            'text': """
            Albert Einstein was a German-born theoretical physicist. Einstein 
            developed the theory of relativity, one of the two pillars of modern 
            physics. He received the 1921 Nobel Prize in Physics for his discovery 
            of the law of the photoelectric effect. Einstein is best known for his 
            mass-energy equivalence formula E=mc². The theory of relativity is a 
            fundamental theory in physics.
            """
        },
        {
            'id': 'quantum_mechanics',
            'text': """
            Quantum mechanics is a fundamental theory in physics. It provides a 
            description of the physical properties of nature at small scales. 
            Niels Bohr was a Danish physicist who made foundational contributions 
            to quantum mechanics. Bohr developed the Bohr model of the atom. 
            Werner Heisenberg was a German theoretical physicist. Heisenberg 
            developed the uncertainty principle, a key principle in quantum mechanics.
            """
        }
    ]
    
    # Build KG
    builder = KnowledgeGraphBuilder()
    result = builder.build_from_documents(documents)
    
    # Save results
    output_dir = 'kg_output_demo'
    builder.save(result, output_dir)
    
    # Display some results
    print("\n📊 Statistics:")
    print(f"  Documents: {result['num_documents']}")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  Entities: {result['num_entities']}")
    print(f"  Triples: {result['num_triples']}")
    print(f"  Taxonomy Nodes: {result['num_nodes']}")
    
    print("\n🔍 Sample Entities:")
    for i, (name, entity) in enumerate(list(result['entities'].items())[:5], 1):
        print(f"  {i}. {name} ({entity.entity_type})")
        if entity.aliases:
            print(f"     Aliases: {', '.join(list(entity.aliases)[:3])}")
    
    print("\n🔗 Sample Triples:")
    for i, triple in enumerate(result['triples'][:10], 1):
        print(f"  {i}. ({triple['head']}) --[{triple['relation']}]--> ({triple['tail']})")
    
    print(f"\n✅ Complete! Results saved to: {output_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    demo()
