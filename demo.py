"""
Example Workflow - Demonstrate LeanRAG-MM capabilities

This example shows:
1. Loading sample data
2. Building taxonomy
3. Running queries
4. Comparing with baseline approaches
"""

import json
import time
from taxonomy_builder import TaxonomyBuilder
from sparse_table import EulerTourLCA
from wu_palmer import WuPalmerSimilarity
from lca_retrieval import LCABoundedRetrieval


def create_sample_triples():
    """Create sample knowledge graph triples."""
    return [
        # Scientific entities
        {"head": "Einstein", "relation": "is_a", "tail": "Scientist", "head_type": "Person", "tail_type": "Concept"},
        {"head": "Scientist", "relation": "is_a", "tail": "Person", "head_type": "Concept", "tail_type": "Concept"},
        {"head": "Person", "relation": "is_a", "tail": "Entity", "head_type": "Concept", "tail_type": "Concept"},
        
        {"head": "Bohr", "relation": "is_a", "tail": "Scientist", "head_type": "Person", "tail_type": "Concept"},
        {"head": "Heisenberg", "relation": "is_a", "tail": "Scientist", "head_type": "Person", "tail_type": "Concept"},
        
        # Relations between entities
        {"head": "Einstein", "relation": "developed", "tail": "Relativity", "head_type": "Person", "tail_type": "Theory"},
        {"head": "Einstein", "relation": "contributed_to", "tail": "Quantum_Mechanics", "head_type": "Person", "tail_type": "Theory"},
        {"head": "Bohr", "relation": "developed", "tail": "Atomic_Model", "head_type": "Person", "tail_type": "Theory"},
        {"head": "Heisenberg", "relation": "developed", "tail": "Uncertainty_Principle", "head_type": "Person", "tail_type": "Theory"},
        
        # Theories
        {"head": "Relativity", "relation": "is_a", "tail": "Physics_Theory", "head_type": "Theory", "tail_type": "Concept"},
        {"head": "Quantum_Mechanics", "relation": "is_a", "tail": "Physics_Theory", "head_type": "Theory", "tail_type": "Concept"},
        {"head": "Atomic_Model", "relation": "is_a", "tail": "Physics_Theory", "head_type": "Theory", "tail_type": "Concept"},
        {"head": "Physics_Theory", "relation": "is_a", "tail": "Theory", "head_type": "Concept", "tail_type": "Concept"},
        {"head": "Theory", "relation": "is_a", "tail": "Concept", "head_type": "Concept", "tail_type": "Concept"},
        
        # Multimodal example
        {"head": "Figure_1", "relation": "illustrates", "tail": "Atomic_Model", "head_type": "Media", "tail_type": "Theory", 
         "head_modality": "image", "tail_modality": "text"},
        {"head": "Table_1", "relation": "contains_data", "tail": "Experiment_Results", "head_type": "Media", "tail_type": "Data",
         "head_modality": "table", "tail_modality": "text"},
    ]


def demo_taxonomy_building():
    """Demonstrate taxonomy building."""
    print("\n" + "="*60)
    print("DEMO 1: Taxonomy Building")
    print("="*60)
    
    triples = create_sample_triples()
    print(f"\n📊 Input: {len(triples)} triples")
    
    # Build taxonomy
    start_time = time.time()
    builder = TaxonomyBuilder()
    taxonomy = builder.build_from_triples(triples)
    build_time = time.time() - start_time
    
    print(f"⏱️  Build time: {build_time*1000:.2f}ms")
    print(f"🌳 Taxonomy nodes: {len(taxonomy.nodes)}")
    print(f"📏 Max depth: {max(node.depth for node in taxonomy.nodes.values())}")
    
    # Show hierarchy
    print("\n📋 Taxonomy Structure:")
    for node in sorted(taxonomy.nodes.values(), key=lambda n: (n.depth, n.name)):
        indent = "  " * node.depth
        print(f"{indent}├─ {node.name} (depth={node.depth}, type={node.node_type})")
    
    return taxonomy


def demo_lca_queries(taxonomy):
    """Demonstrate O(1) LCA queries."""
    print("\n" + "="*60)
    print("DEMO 2: O(1) LCA Queries")
    print("="*60)
    
    # Build LCA structure
    start_time = time.time()
    lca_solver = EulerTourLCA(taxonomy)
    lca_solver.build()
    build_time = time.time() - start_time
    
    print(f"\n⏱️  LCA structure build time: {build_time*1000:.2f}ms")
    print(f"📊 Euler tour length: {len(lca_solver.euler_tour)}")
    
    # Test queries
    test_pairs = [
        ("Einstein", "Bohr"),
        ("Einstein", "Quantum_Mechanics"),
        ("Relativity", "Atomic_Model"),
    ]
    
    print("\n🔍 LCA Query Results:")
    for name1, name2 in test_pairs:
        node1_id = taxonomy.name_to_id.get(name1)
        node2_id = taxonomy.name_to_id.get(name2)
        
        if node1_id and node2_id:
            start = time.time()
            lca_id = lca_solver.query(node1_id, node2_id)
            query_time = time.time() - start
            
            lca_node = taxonomy.nodes[lca_id]
            print(f"\n  {name1} ∩ {name2}:")
            print(f"    LCA: {lca_node.name}")
            print(f"    Depth: {lca_node.depth}")
            print(f"    Query time: {query_time*1e6:.1f}µs (microseconds!)")
    
    return lca_solver


def demo_wu_palmer_similarity(taxonomy, lca_solver):
    """Demonstrate Wu-Palmer similarity."""
    print("\n" + "="*60)
    print("DEMO 3: Wu-Palmer Similarity")
    print("="*60)
    
    wp = WuPalmerSimilarity(lca_solver)
    
    test_pairs = [
        ("Einstein", "Bohr", "Same category (both Scientists)"),
        ("Einstein", "Relativity", "Person vs their work"),
        ("Relativity", "Quantum_Mechanics", "Related theories"),
        ("Einstein", "Figure_1", "Cross-modal: Person vs Media"),
    ]
    
    print("\n📏 Similarity Scores:")
    for name1, name2, description in test_pairs:
        node1_id = taxonomy.name_to_id.get(name1)
        node2_id = taxonomy.name_to_id.get(name2)
        
        if node1_id and node2_id:
            similarity = wp.similarity(node1_id, node2_id)
            print(f"\n  {name1} ↔ {name2}")
            print(f"    Description: {description}")
            print(f"    Wu-Palmer: {similarity:.3f}")
            print(f"    Interpretation: {'Highly related' if similarity > 0.7 else 'Moderately related' if similarity > 0.4 else 'Distantly related'}")


def demo_lca_bounded_retrieval(taxonomy, lca_solver):
    """Demonstrate LCA-bounded retrieval."""
    print("\n" + "="*60)
    print("DEMO 4: LCA-Bounded Retrieval")
    print("="*60)
    
    retrieval = LCABoundedRetrieval(taxonomy, lca_solver)
    
    queries = [
        "What did Einstein work on?",
        "Quantum mechanics theories",
    ]
    
    for query in queries:
        print(f"\n🔎 Query: '{query}'")
        
        start_time = time.time()
        results = retrieval.retrieve(query, top_k=5, min_similarity=0.3)
        retrieval_time = time.time() - start_time
        
        print(f"⏱️  Retrieval time: {retrieval_time*1000:.2f}ms")
        print(f"📋 Results ({len(results)} found):")
        
        for i, result in enumerate(results[:5], 1):
            print(f"\n  {i}. {result.name}")
            print(f"     Similarity: {result.similarity:.3f}")
            print(f"     Type: {result.node_type} ({result.modality})")
            print(f"     Depth: {result.depth}")


def main():
    """Run all demos."""
    print("\n" + "🚀"*30)
    print(" "*20 + "LeanRAG-MM Demo")
    print("🚀"*30)
    
    # Demo 1: Build taxonomy
    taxonomy = demo_taxonomy_building()
    
    # Demo 2: LCA queries
    lca_solver = demo_lca_queries(taxonomy)
    
    # Demo 3: Wu-Palmer similarity
    demo_wu_palmer_similarity(taxonomy, lca_solver)
    
    # Demo 4: LCA-bounded retrieval
    demo_lca_bounded_retrieval(taxonomy, lca_solver)
    
    print("\n" + "="*60)
    print("✅ All demos completed successfully!")
    print("="*60)
    print("\n📚 Next steps:")
    print("  1. Try with your own data: python pipeline.py --mode build --input your_data.jsonl")
    print("  2. Run queries: python pipeline.py --mode query --query 'your question'")
    print("  3. Read the architecture docs: # 🏗️ LeanRAG-MM Architecture.md")
    print()


if __name__ == "__main__":
    main()
