# VATRAG 2.0 - Quick Reference Guide

## 🚀 Quick Start (3 Commands)

```bash
# 1. Run demo to see it work
python3 example_workflow.py

# 2. Build from VATRAG data
python3 integrate_vatrag.py --vatrag-data ../VATRAG/ckg_data/mix_chunk

# 3. Query your data
python3 pipeline.py --mode query --query "your question here"
```

## 📚 Core Concepts

### Hierarchical Knowledge Graph
- **Nodes**: Entities organized in tree structure
- **Root**: Virtual root connecting all components
- **Depth**: Distance from root (0 = root)
- **LCA**: Lowest Common Ancestor (nearest shared parent)

### Wu-Palmer Similarity
```
wu_palmer(u, v) = 2 × depth(lca(u,v)) / (depth(u) + depth(v))

Range: [0, 1]
1.0 = identical nodes
0.0 = no common ancestor
```

### Retrieval Strategies

| Strategy | Threshold | Use Case |
|----------|-----------|----------|
| `strict` | 0.8 | Very close entities only |
| `moderate` | 0.5 | Same category (default) |
| `loose` | 0.3 | Related domain |
| `exploratory` | 0.1 | Any connection |

## 🔧 Common Tasks

### Build Taxonomy from Triples

```python
from taxonomy_builder import TaxonomyBuilder

# Load triples
triples = [
    {'head': 'Einstein', 'relation': 'is_a', 'tail': 'Physicist', ...},
    # ... more triples
]

# Build
taxonomy = TaxonomyBuilder()
root_id = taxonomy.build_from_triples(triples)

# Save
taxonomy.save('my_taxonomy.json')

# Print
taxonomy.print_tree(max_depth=3)
```

### Build LCA Structure

```python
from sparse_table import EulerTourLCA

# Build
lca_solver = EulerTourLCA()
tree = taxonomy.get_tree_adjacency()
depths = taxonomy.get_node_depths()
lca_solver.build(tree, root_id, depths)

# Save
lca_solver.save('lca_structure.pkl')

# Query - O(1)!
lca_node = lca_solver.lca(node_u, node_v)
```

### Compute Similarity

```python
from wu_palmer import WuPalmerSimilarity

# Initialize
wp = WuPalmerSimilarity(lca_solver)

# Single pair - O(1)
similarity = wp.similarity(node_u, node_v)

# Batch
results = wp.batch_similarity(query_node, [candidate1, candidate2, ...])
# Returns: [(node_id, similarity), ...] sorted by similarity
```

### Retrieve with LCA-Bounded Search

```python
from lca_retrieval import LCABoundedRetrieval

# Initialize
retriever = LCABoundedRetrieval(taxonomy, lca_solver, wp)

# Query
results = retriever.retrieve(
    query="Einstein quantum mechanics",
    threshold=0.5,  # or use strategy='moderate'
    top_k=20
)

# Get context
context = retriever.assemble_hierarchical_context(results)

# Explain why retrieved
explanation = retriever.explain_retrieval(query, results[0])
```

### Process Multimodal Documents

```python
from multimodal_extractor import MultimodalKGBuilder

# Prepare documents
documents = [{
    'source_id': 'doc1',
    'text': 'Einstein published work on photoelectric effect...',
    'images': [
        {'path': 'fig1.png', 'caption': 'Apparatus diagram...'}
    ],
    'tables': [
        {'headers': ['Exp', 'Energy'], 'rows': [['1', '3.1eV'], ...]}
    ]
}]

# Build
builder = MultimodalKGBuilder()
taxonomy, triples = builder.build_unified_taxonomy(documents, 'output_dir')
```

## 📊 Performance Tips

### For Large Graphs (>10K nodes)

1. **Use batch operations**
   ```python
   # Instead of:
   for node in nodes:
       sim = wp.similarity(query, node)
   
   # Use:
   results = wp.batch_similarity(query, nodes)
   ```

2. **Save/load precomputed structures**
   ```python
   # Build once
   taxonomy.save('taxonomy.json')
   lca_solver.save('lca.pkl')
   
   # Load many times
   taxonomy.load('taxonomy.json')
   lca_solver.load('lca.pkl')
   ```

3. **Adjust max_depth_expansion**
   ```python
   # Faster, less comprehensive
   results = retriever.retrieve(query, top_k=20, max_depth_expansion=2)
   
   # Slower, more comprehensive
   results = retriever.retrieve(query, top_k=20, max_depth_expansion=5)
   ```

## 🐛 Troubleshooting

### "No entities found in query"
- Query uses keyword matching by default
- Add more entities to taxonomy or use broader keywords
- Try `exploratory` strategy for fallback search

### "Module not found"
```bash
pip3 install numpy pyyaml
```

### Slow retrieval
- Check max_depth_expansion (reduce if too high)
- Verify LCA structure is loaded (not rebuilt each time)
- Use stricter threshold to prune more aggressively

### Empty results
- Check if query entities exist in taxonomy
- Try looser strategy (`loose` or `exploratory`)
- Verify taxonomy was built correctly (`taxonomy.print_tree()`)

## 📈 Comparison Cheat Sheet

| Aspect | Original LeanRAG | VATRAG 2.0 |
|--------|-----------------|------------|
| **Build** | `build_graph.py` (30 min) | `taxonomy_builder.py` (<1s) |
| **Hierarchy** | Louvain (random) | Taxonomy (deterministic) |
| **Similarity** | Cosine O(1536) | Wu-Palmer O(1) |
| **Retrieval** | Milvus + BM25 O(n) | LCA-bounded O(k) |
| **Storage** | 14.5 MB | 0.5 MB |
| **API** | Required ($$$) | None |

## 🎯 Use Cases

### 1. Research Paper QA
```bash
# Build from paper triples (text + figures + tables)
python3 pipeline.py --mode build --input papers/triples.jsonl

# Query across modalities
python3 pipeline.py --mode query \
  --query "What experiments support the theory?" \
  --strategy moderate
```

### 2. Domain Knowledge Base
```bash
# Medical/legal/technical KG
python3 pipeline.py --mode build --input domain/triples.jsonl

# Hierarchical retrieval respects taxonomy
python3 pipeline.py --mode query \
  --query "treatment for condition X" \
  --strategy strict
```

### 3. Cross-Modal Search
```python
# Text query → text + image + table results
results = retriever.retrieve("climate change impacts")

for r in results:
    if r.modality == 'image':
        print(f"Figure: {r.name}")
    elif r.modality == 'table':
        print(f"Data: {r.name}")
```

## 📝 File Formats

### Triple Format (JSONL)
```json
{
  "head": "Einstein",
  "head_type": "Person",
  "head_description": "Albert Einstein, physicist",
  "head_modality": "text",
  "relation": "is_a",
  "tail": "Physicist",
  "tail_type": "Profession",
  "tail_description": "Scientist specializing in physics",
  "tail_modality": "text",
  "source_id": "doc1"
}
```

### Taxonomy Format (JSON)
```json
{
  "root_id": 0,
  "nodes": {
    "0": {
      "name": "ROOT",
      "type": "VirtualRoot",
      "modality": "text",
      "depth": 0,
      "parent_id": null,
      "children": [1, 2],
      "description": "...",
      "source_ids": []
    }
  },
  "name_to_id": {"ROOT": 0, ...}
}
```

## 🔗 API Reference

See individual files for detailed API:
- `sparse_table.py` - LCA implementation
- `wu_palmer.py` - Similarity computation
- `taxonomy_builder.py` - Hierarchy construction
- `lca_retrieval.py` - Retrieval algorithm
- `multimodal_extractor.py` - Multimodal support

## 💡 Tips & Tricks

1. **Start small** - Test with demo before real data
2. **Use strategies** - Let threshold adapt to your use case
3. **Check explanations** - Use `explain_retrieval()` to debug
4. **Visualize taxonomy** - Use `print_tree()` to understand structure
5. **Save everything** - Precompute once, query many times

## 🤝 Integration with VATRAG

```bash
# Use VATRAG for chunking + triple extraction
cd ../VATRAG
./run_file_chunk.sh

# Use VATRAG2.0 for taxonomy + retrieval
cd ../VATRAG2.0
python3 integrate_vatrag.py --vatrag-data ../VATRAG/ckg_data/mix_chunk
```

---

**For more details, see:**
- `README.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `example_workflow.py` - Working examples
