#!/usr/bin/env python3
"""
Integration Script - Build VATRAG 2.0 from VATRAG Output

This script shows how to:
1. Use VATRAG's chunking and triple extraction
2. Build hierarchical taxonomy with LCA
3. Compare retrieval performance

Usage:
    python integrate_vatrag.py --vatrag-data ../VATRAG/ckg_data/mix_chunk
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add VATRAG to path
vatrag_path = Path(__file__).parent.parent / 'VATRAG'
sys.path.insert(0, str(vatrag_path))

from pipeline import HierarchicalKGPipeline


def find_vatrag_data(data_dir: str) -> tuple:
    """
    Find entity.jsonl and relation.jsonl in VATRAG output directory.
    
    Returns:
        tuple: (entity_file_path, relation_file_path)
    """
    data_path = Path(data_dir)
    
    entity_file = data_path / 'entity.jsonl'
    relation_file = data_path / 'relation.jsonl'
    
    if not entity_file.exists():
        raise FileNotFoundError(f"Entity file not found: {entity_file}")
    if not relation_file.exists():
        raise FileNotFoundError(f"Relation file not found: {relation_file}")
    
    return str(entity_file), str(relation_file)


def load_vatrag_entities(data_dir: str):
    """Load VATRAG entity file for comparison."""
    entity_file = Path(data_dir) / 'entity.jsonl'
    
    if not entity_file.exists():
        print(f"Warning: {entity_file} not found")
        return []
    
    entities = []
    with open(entity_file, 'r') as f:
        for line in f:
            entities.append(json.loads(line))
    
    return entities


def compare_with_original(vatrag_entities, taxonomy_nodes):
    """Compare VATRAG 2.0 with original VATRAG."""
    print("\n" + "="*70)
    print("Comparison: Original VATRAG vs VATRAG 2.0")
    print("="*70)
    
    # Count entities
    vatrag_count = len(vatrag_entities)
    vatrag2_count = len(taxonomy_nodes)
    
    print(f"\nEntity Count:")
    print(f"  Original VATRAG: {vatrag_count}")
    print(f"  VATRAG 2.0:      {vatrag2_count}")
    
    # Storage comparison (rough estimate)
    # Original: 1536-dim embeddings per entity
    vatrag_storage = vatrag_count * 1536 * 4  # 4 bytes per float32
    # VATRAG 2.0: 16 bytes per node
    vatrag2_storage = vatrag2_count * 16
    
    print(f"\nEstimated Storage:")
    print(f"  Original VATRAG: {vatrag_storage / 1024:.1f} KB (embeddings)")
    print(f"  VATRAG 2.0:      {vatrag2_storage / 1024:.1f} KB (taxonomy)")
    print(f"  Reduction:       {vatrag_storage / vatrag2_storage:.1f}×")
    
    # Hierarchy info
    max_depth = max(node.depth for node in taxonomy_nodes.values())
    avg_depth = sum(node.depth for node in taxonomy_nodes.values()) / len(taxonomy_nodes)
    
    print(f"\nHierarchy (VATRAG 2.0):")
    print(f"  Max Depth:  {max_depth}")
    print(f"  Avg Depth:  {avg_depth:.2f}")
    
    # Depth distribution
    depth_dist = {}
    for node in taxonomy_nodes.values():
        depth_dist[node.depth] = depth_dist.get(node.depth, 0) + 1
    
    print(f"\n  Depth Distribution:")
    for depth in sorted(depth_dist.keys())[:6]:  # First 6 levels
        count = depth_dist[depth]
        bar = '█' * (count // 10)
        print(f"    Level {depth}: {count:4d} nodes {bar}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Integrate VATRAG output with VATRAG 2.0'
    )
    parser.add_argument(
        '--vatrag-data',
        type=str,
        default='../VATRAG/ckg_data/mix_chunk',
        help='Path to VATRAG data directory (containing triple files)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='taxonomy_output',
        help='Output directory for VATRAG 2.0 taxonomy'
    )
    parser.add_argument(
        '--test-queries',
        nargs='+',
        help='Test queries to run after building'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("VATRAG → VATRAG 2.0 Integration")
    print("="*70)
    
    # Find entity and relation files
    print(f"\nSearching for VATRAG data in {args.vatrag_data}...")
    try:
        entity_file, relation_file = find_vatrag_data(args.vatrag_data)
        print(f"Found entity file: {entity_file}")
        print(f"Found relation file: {relation_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you've run VATRAG pipeline first:")
        print("  cd ../VATRAG")
        print("  ./run_file_chunk.sh")
        print("\nOr check that entity.jsonl and relation.jsonl exist in:")
        print(f"  {args.vatrag_data}")
        return 1
    
    # Load VATRAG entities for comparison
    print("\nLoading original VATRAG entities...")
    vatrag_entities = load_vatrag_entities(args.vatrag_data)
    print(f"Loaded {len(vatrag_entities)} entities")
    
    # Build VATRAG 2.0 taxonomy
    print("\nBuilding VATRAG 2.0 taxonomy...")
    pipeline = HierarchicalKGPipeline()
    
    start_time = time.time()
    pipeline.build_from_vatrag_data(args.vatrag_data, args.output)
    build_time = time.time() - start_time
    
    print(f"\n✓ Build completed in {build_time:.2f}s")
    
    # Compare
    if vatrag_entities and pipeline.taxonomy:
        compare_with_original(vatrag_entities, pipeline.taxonomy.nodes)
    
    # Test queries
    if args.test_queries:
        print("\n" + "="*70)
        print("Testing Queries")
        print("="*70)
        
        for query in args.test_queries:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print('='*70)
            
            start_time = time.time()
            results, context = pipeline.query(query, strategy='moderate')
            query_time = time.time() - start_time
            
            print(f"\nRetrieved {len(results)} results in {query_time*1000:.2f}ms")
            print("\nTop 5 Results:")
            for i, result in enumerate(results[:5], 1):
                print(f"  {i}. {result.name} (similarity={result.similarity:.3f}, "
                      f"type={result.node_type}, depth={result.depth})")
    
    # Summary
    print("\n" + "="*70)
    print("Integration Complete!")
    print("="*70)
    print(f"\nOutput saved to: {args.output}/")
    print(f"  - taxonomy.json       : Hierarchical taxonomy")
    print(f"  - lca_structure.pkl   : LCA query structure")
    print("\nNext steps:")
    print(f"  1. Query: python pipeline.py --mode query --query 'your question'")
    print(f"  2. View taxonomy: cat {args.output}/taxonomy.json | jq '.nodes | length'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
