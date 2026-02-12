"""
Main Pipeline - Hierarchical KG with LCA-Based Retrieval

This is the main entry point that integrates:
1. Chunking (from original VATRAG)
2. Triple extraction (from original VATRAG) 
3. Taxonomy building (NEW - replaces Louvain)
4. LCA structure construction (NEW - O(1) queries)
5. Wu-Palmer similarity (NEW - replaces cosine)
6. LCA-bounded retrieval (NEW - replaces Milvus+BM25)
7. Multimodal support (NEW)

Usage:
    python pipeline.py --mode build --input ckg_data/mix_chunk
    python pipeline.py --mode query --query "How did Einstein influence quantum mechanics?"
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

# Add parent directory to path to import from VATRAG
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VATRAG'))

from taxonomy_builder import TaxonomyBuilder
from sparse_table import EulerTourLCA
from wu_palmer import WuPalmerSimilarity
from lca_retrieval import LCABoundedRetrieval
from multimodal_extractor import MultimodalKGBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HierarchicalKGPipeline:
    """
    Main pipeline for hierarchical KG with LCA-based retrieval.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.taxonomy = None
        self.lca_solver = None
        self.wp_calculator = None
        self.retriever = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        import yaml
        
        # Try to load from VATRAG2.0 first, then parent VATRAG
        if os.path.exists(config_path):
            config_file = config_path
        else:
            parent_config = os.path.join('..', 'VATRAG', 'config.yaml')
            if os.path.exists(parent_config):
                config_file = parent_config
            else:
                logger.warning("No config file found, using defaults")
                return self._default_config()
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_file}")
        return config
    
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'taxonomy': {
                'virtual_root': 'ROOT',
                'output_dir': 'taxonomy_output'
            },
            'retrieval': {
                'default_threshold': 0.5,
                'default_strategy': 'moderate',
                'top_k': 20,
                'max_depth_expansion': 3
            },
            'multimodal': {
                'enabled': True,
                'image_captions': True,
                'table_extraction': True
            }
        }
    
    def build_from_vatrag_data(self, data_dir: str, output_dir: str):
        """
        Build taxonomy from VATRAG data directory (entity.jsonl + relation.jsonl).
        
        Args:
            data_dir: Path to VATRAG data directory containing entity.jsonl and relation.jsonl
            output_dir: Output directory for taxonomy
        """
        logger.info(f"Building taxonomy from VATRAG data in {data_dir}")
        start_time = time.time()
        
        # Load entities
        entity_file = os.path.join(data_dir, 'entity.jsonl')
        relation_file = os.path.join(data_dir, 'relation.jsonl')
        
        if not os.path.exists(entity_file):
            raise FileNotFoundError(f"Entity file not found: {entity_file}")
        if not os.path.exists(relation_file):
            raise FileNotFoundError(f"Relation file not found: {relation_file}")
        
        # Load entities into a dict for lookup
        entities = {}
        with open(entity_file, 'r') as f:
            for line in f:
                if line.strip():
                    entity = json.loads(line)
                    entities[entity['entity_name']] = entity
        
        logger.info(f"Loaded {len(entities)} entities")
        
        # Load relations and convert to triples
        triples = []
        with open(relation_file, 'r') as f:
            for line in f:
                if line.strip():
                    relation = json.loads(line)
                    
                    # Get entity info
                    head_name = relation.get('src_tgt', '')
                    tail_name = relation.get('tgt_src', '')
                    relation_type = relation.get('source', 'related_to')
                    
                    # Get entity details
                    head_entity = entities.get(head_name, {})
                    tail_entity = entities.get(tail_name, {})
                    
                    triple = {
                        'head': head_name,
                        'head_description': head_entity.get('description', f'Entity: {head_name}'),
                        'head_type': head_entity.get('type', 'Unknown'),
                        'head_modality': 'text',
                        'relation': relation_type,
                        'tail': tail_name,
                        'tail_description': tail_entity.get('description', f'Entity: {tail_name}'),
                        'tail_type': tail_entity.get('type', 'Unknown'),
                        'tail_modality': 'text',
                        'source_id': relation.get('source_id', '')
                    }
                    triples.append(triple)
        
        logger.info(f"Loaded {len(triples)} relations/triples")
        
        # Add entity-only triples for entities not in relations
        # (These create IS-A relations from entity type)
        entity_triples = []
        entities_in_relations = set()
        for triple in triples:
            entities_in_relations.add(triple['head'])
            entities_in_relations.add(triple['tail'])
        
        for entity_name, entity_data in entities.items():
            if entity_name not in entities_in_relations:
                # Create IS-A relation from entity to its type
                entity_type = entity_data.get('type', 'Unknown')
                if entity_type and entity_type != 'Unknown':
                    entity_triples.append({
                        'head': entity_name,
                        'head_description': entity_data.get('description', f'Entity: {entity_name}'),
                        'head_type': entity_type,
                        'head_modality': 'text',
                        'relation': 'is_a',
                        'tail': entity_type,
                        'tail_description': f'Category: {entity_type}',
                        'tail_type': 'Category',
                        'tail_modality': 'text',
                        'source_id': entity_data.get('source_id', '')
                    })
        
        all_triples = triples + entity_triples
        logger.info(f"Total triples (relations + entity types): {len(all_triples)}")
        
        # Build taxonomy
        self.taxonomy = TaxonomyBuilder(
            virtual_root_name=self.config['taxonomy']['virtual_root']
        )
        root_id = self.taxonomy.build_from_triples(all_triples)
        
        # Save taxonomy
        os.makedirs(output_dir, exist_ok=True)
        taxonomy_path = os.path.join(output_dir, 'taxonomy.json')
        self.taxonomy.save(taxonomy_path)
        
        # Build LCA structure
        logger.info("Building LCA structure...")
        self.lca_solver = EulerTourLCA()
        tree = self.taxonomy.get_tree_adjacency()
        node_depths = self.taxonomy.get_node_depths()
        self.lca_solver.build(tree, root_id, node_depths)
        
        # Save LCA structure
        lca_path = os.path.join(output_dir, 'lca_structure.pkl')
        self.lca_solver.save(lca_path)
        
        # Initialize Wu-Palmer calculator
        self.wp_calculator = WuPalmerSimilarity(self.lca_solver)
        
        # Initialize retriever
        self.retriever = LCABoundedRetrieval(
            self.taxonomy,
            self.lca_solver,
            self.wp_calculator
        )
        
        build_time = time.time() - start_time
        
        # Print statistics
        stats = self.lca_solver.stats()
        logger.info("="*70)
        logger.info("Build Complete!")
        logger.info("="*70)
        logger.info(f"Build Time: {build_time:.2f}s")
        logger.info(f"Total Nodes: {stats['num_nodes']}")
        logger.info(f"Max Depth: {stats['max_depth']}")
        logger.info(f"Avg Depth: {stats['avg_depth']:.2f}")
        logger.info(f"Euler Tour Length: {stats['euler_tour_length']}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info("="*70)
        
        # Print sample of taxonomy
        print("\nTaxonomy Sample (first 3 levels):")
        self.taxonomy.print_tree(max_depth=3)
    
    def load_from_disk(self, taxonomy_path: str, lca_path: str):
        """Load pre-built taxonomy and LCA structure from disk."""
        logger.info("Loading taxonomy from disk...")
        
        # Load taxonomy
        self.taxonomy = TaxonomyBuilder()
        self.taxonomy.load(taxonomy_path)
        
        # Load LCA structure
        self.lca_solver = EulerTourLCA()
        self.lca_solver.load(lca_path)
        
        # Initialize Wu-Palmer calculator
        self.wp_calculator = WuPalmerSimilarity(self.lca_solver)
        
        # Initialize retriever
        self.retriever = LCABoundedRetrieval(
            self.taxonomy,
            self.lca_solver,
            self.wp_calculator
        )
        
        logger.info("Loaded successfully!")
    
    def query(self, query_text: str, threshold: float = None, strategy: str = None):
        """
        Execute query using LCA-bounded retrieval.
        
        Args:
            query_text: Query string
            threshold: Wu-Palmer threshold (optional)
            strategy: Threshold strategy (optional)
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Run build or load first.")
        
        # Use config defaults if not specified
        if threshold is None:
            threshold = self.config['retrieval']['default_threshold']
        if strategy is None:
            strategy = self.config['retrieval']['default_strategy']
        
        logger.info("="*70)
        logger.info(f"Query: {query_text}")
        logger.info(f"Strategy: {strategy} (threshold={threshold})")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Retrieve
        results = self.retriever.retrieve(
            query_text,
            threshold=threshold,
            top_k=self.config['retrieval']['top_k'],
            strategy=strategy
        )
        
        query_time = time.time() - start_time
        
        # Assemble context
        context = self.retriever.assemble_hierarchical_context(results)
        
        # Print results
        print(f"\nRetrieved {len(results)} results in {query_time*1000:.2f}ms:\n")
        
        for i, result in enumerate(results[:10], 1):  # Top 10
            print(f"{i}. {result.name}")
            print(f"   Type: {result.node_type} | Modality: {result.modality} | Depth: {result.depth}")
            print(f"   Similarity: {result.similarity:.3f} | Path Length: {result.path_length}")
            print(f"   Description: {result.description[:100]}...")
            
            # Show explanation for top result
            if i == 1:
                explanation = self.retriever.explain_retrieval(query_text, result)
                print(f"\n{explanation}")
            print()
        
        print("="*70)
        print(f"Query Time: {query_time*1000:.2f}ms")
        print("="*70)
        
        return results, context


def main():
    parser = argparse.ArgumentParser(description='Hierarchical KG with LCA-Based Retrieval')
    parser.add_argument('--mode', choices=['build', 'query', 'demo'], required=True,
                       help='Mode: build taxonomy, query, or run demo')
    parser.add_argument('--input', type=str,
                       help='Input path (for build mode: path to VATRAG data directory with entity.jsonl and relation.jsonl)')
    parser.add_argument('--output', type=str, default='taxonomy_output',
                       help='Output directory for taxonomy')
    parser.add_argument('--query', type=str,
                       help='Query string (for query mode)')
    parser.add_argument('--taxonomy', type=str,
                       help='Path to taxonomy.json (for query mode)')
    parser.add_argument('--lca', type=str,
                       help='Path to lca_structure.pkl (for query mode)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Wu-Palmer similarity threshold')
    parser.add_argument('--strategy', type=str, default='moderate',
                       choices=['strict', 'moderate', 'loose', 'exploratory'],
                       help='Retrieval strategy')
    
    args = parser.parse_args()
    
    pipeline = HierarchicalKGPipeline()
    
    if args.mode == 'build':
        if not args.input:
            parser.error("--input required for build mode")
        
        pipeline.build_from_vatrag_data(args.input, args.output)
    
    elif args.mode == 'query':
        if not args.query:
            parser.error("--query required for query mode")
        
        # Load from disk
        if args.taxonomy and args.lca:
            pipeline.load_from_disk(args.taxonomy, args.lca)
        else:
            # Try to load from default output directory
            taxonomy_path = os.path.join(args.output, 'taxonomy.json')
            lca_path = os.path.join(args.output, 'lca_structure.pkl')
            
            if not os.path.exists(taxonomy_path) or not os.path.exists(lca_path):
                parser.error("Taxonomy not found. Run build mode first or specify --taxonomy and --lca paths")
            
            pipeline.load_from_disk(taxonomy_path, lca_path)
        
        # Execute query
        pipeline.query(args.query, threshold=args.threshold, strategy=args.strategy)
    
    elif args.mode == 'demo':
        print("Running Demo...")
        print("="*70)
        
        # Create demo data
        demo_triples = [
            {'head': 'Einstein', 'head_type': 'Person', 'head_description': 'Albert Einstein, physicist',
             'relation': 'is_a', 'tail': 'Scientist', 'tail_type': 'Concept', 'tail_description': 'A person who conducts scientific research'},
            {'head': 'Bohr', 'head_type': 'Person', 'head_description': 'Niels Bohr, physicist',
             'relation': 'is_a', 'tail': 'Scientist', 'tail_type': 'Concept', 'tail_description': 'A person who conducts scientific research'},
            {'head': 'Scientist', 'head_type': 'Concept', 'head_description': 'A person who conducts scientific research',
             'relation': 'is_a', 'tail': 'Person', 'tail_type': 'Concept', 'tail_description': 'A human being'},
            {'head': 'Quantum Mechanics', 'head_type': 'Concept', 'head_description': 'Branch of physics',
             'relation': 'is_a', 'tail': 'Physics', 'tail_type': 'Concept', 'tail_description': 'Natural science'},
            {'head': 'Einstein', 'head_type': 'Person', 'head_description': 'Albert Einstein',
             'relation': 'contributed_to', 'tail': 'Quantum Mechanics', 'tail_type': 'Concept', 'tail_description': 'Branch of physics'},
        ]
        
        # Save demo triples
        demo_path = '/tmp/demo_triples.jsonl'
        with open(demo_path, 'w') as f:
            for triple in demo_triples:
                # Convert to VATRAG format
                triple_str = f"<{triple['head']}>\t<{triple['head_description']}>\t<{triple['head_type']}>\t<{triple['relation']}>\t<>\t<{triple['tail']}>\t<{triple['tail_description']}>\t<{triple['tail_type']}>"
                f.write(json.dumps({'triple': triple_str, 'source_id': 'demo'}) + '\n')
        
        # Build
        pipeline.build_from_vatrag_triples(demo_path, '/tmp/demo_taxonomy')
        
        # Query
        print("\n\nDemo Query:")
        pipeline.query("Einstein quantum mechanics", strategy='moderate')


if __name__ == "__main__":
    main()
