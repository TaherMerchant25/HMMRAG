"""
LCA-Bounded Retrieval Algorithm with Wu-Palmer Similarity

This module implements the core novelty: retrieval using LCA-bounded
subtree search with Wu-Palmer pruning, replacing flat Milvus+BM25 search.

Key Features:
- O(k) search complexity vs O(n) for flat retrieval
- Subtree pruning based on Wu-Palmer thresholds
- Hierarchical context assembly (deep → broad)
- Cross-modal entity fusion
- Deterministic and interpretable results

Comparison with Original retrieve.py:
┌─────────────────────┬──────────────────┬────────────────────┐
│ Aspect              │ Original         │ LCA-Bounded (Ours) │
├─────────────────────┼──────────────────┼────────────────────┤
│ Search Complexity   │ O(n)             │ O(k), k << n       │
│ Similarity Metric   │ Cosine O(1536)   │ Wu-Palmer O(1)     │
│ Dependencies        │ Milvus + API     │ None               │
│ Pruning Strategy    │ Top-k only       │ Subtree pruning    │
│ Interpretability    │ Low (embeddings) │ High (LCA path)    │
└─────────────────────┴──────────────────┴────────────────────┘
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    node_id: int
    name: str
    description: str
    node_type: str
    modality: str
    depth: int
    similarity: float
    path_length: int
    lca_node: int
    
    def __repr__(self):
        return (f"RetrievalResult(name='{self.name}', "
                f"similarity={self.similarity:.3f}, "
                f"depth={self.depth}, "
                f"type={self.node_type})")


class LCABoundedRetrieval:
    """
    LCA-bounded retrieval with Wu-Palmer similarity and subtree pruning.
    
    Algorithm:
    1. Extract query entities (NER or keyword matching)
    2. For each query entity:
       a. Start from parent in taxonomy
       b. Search siblings first (highest similarity expected)
       c. Compute Wu-Palmer similarity with candidates
       d. If Wu-Palmer < threshold, PRUNE entire subtree
       e. Expand upward in tree if needed
    3. Merge results from all query entities
    4. Assemble hierarchical context (deep → broad)
    """
    
    def __init__(self, taxonomy_builder, lca_solver, wu_palmer_calculator):
        """
        Initialize LCA-bounded retrieval.
        
        Args:
            taxonomy_builder: TaxonomyBuilder instance
            lca_solver: EulerTourLCA instance
            wu_palmer_calculator: WuPalmerSimilarity instance
        """
        self.taxonomy = taxonomy_builder
        self.lca = lca_solver
        self.wp = wu_palmer_calculator
        
        # Build reverse index: name → node_id (for entity lookup)
        self.name_index = {}
        for nid, node in self.taxonomy.nodes.items():
            # Store both exact and lowercase
            self.name_index[node.name] = nid
            self.name_index[node.name.lower()] = nid
    
    def extract_query_entities(self, query: str) -> List[int]:
        """
        Extract entities from query string.
        
        For now, uses simple keyword matching against entity names.
        Can be enhanced with NER (spaCy, etc.) in the future.
        
        Args:
            query: Query string
        
        Returns:
            List of node IDs found in query
        """
        query_lower = query.lower()
        found_entities = []
        
        # Sort by length descending to match longer phrases first
        sorted_names = sorted(self.name_index.keys(), key=len, reverse=True)
        
        for name in sorted_names:
            if name.lower() in query_lower:
                node_id = self.name_index[name]
                if node_id not in found_entities:
                    found_entities.append(node_id)
                    # Remove matched text to avoid duplicate matches
                    query_lower = query_lower.replace(name.lower(), '', 1)
        
        logger.info(f"Extracted {len(found_entities)} entities from query")
        return found_entities
    
    def search_from_node(self, query_node_id: int, 
                         threshold: float = 0.5,
                         top_k: int = 20,
                         max_depth_expansion: int = 3) -> List[RetrievalResult]:
        """
        Search for similar nodes starting from query node with LCA-bounded expansion.
        
        Args:
            query_node_id: Starting node ID
            threshold: Wu-Palmer similarity threshold
            top_k: Maximum number of results
            max_depth_expansion: How many levels up to expand search
        
        Returns:
            List of RetrievalResult, sorted by similarity descending
        """
        results = []
        visited = set([query_node_id])
        query_node = self.taxonomy.nodes[query_node_id]
        
        # Start from parent and expand upward
        current_search_root = query_node.parent_id
        expansion_level = 0
        
        while (len(results) < top_k and 
               current_search_root is not None and 
               expansion_level <= max_depth_expansion):
            
            # Get all nodes in subtree of current search root
            subtree_nodes = self._get_subtree_nodes(current_search_root)
            
            for candidate_id in subtree_nodes:
                if candidate_id in visited:
                    continue
                
                visited.add(candidate_id)
                candidate_node = self.taxonomy.nodes[candidate_id]
                
                # Compute Wu-Palmer similarity
                similarity = self.wp.similarity(query_node_id, candidate_id)
                
                if similarity >= threshold:
                    # Compute additional metrics
                    lca_node = self.lca.lca(query_node_id, candidate_id)
                    path_length = self.wp.path_length(query_node_id, candidate_id)
                    
                    result = RetrievalResult(
                        node_id=candidate_id,
                        name=candidate_node.name,
                        description=candidate_node.description,
                        node_type=candidate_node.node_type,
                        modality=candidate_node.modality,
                        depth=candidate_node.depth,
                        similarity=similarity,
                        path_length=path_length,
                        lca_node=lca_node
                    )
                    results.append(result)
                elif candidate_node.children:
                    # Prune: if parent has low similarity, skip all children
                    visited.update(self._get_subtree_nodes(candidate_id))
            
            # Expand search upward
            if len(results) < top_k:
                parent = self.taxonomy.nodes[current_search_root].parent_id
                if parent is not None:
                    current_search_root = parent
                    expansion_level += 1
                else:
                    break
        
        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)
        
        logger.info(f"Found {len(results)} results for node {query_node.name} "
                   f"(threshold={threshold}, expansions={expansion_level})")
        
        return results[:top_k]
    
    def _get_subtree_nodes(self, root_id: int) -> List[int]:
        """Get all node IDs in subtree rooted at root_id (BFS)."""
        if root_id not in self.taxonomy.nodes:
            return []
        
        result = []
        queue = [root_id]
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            node = self.taxonomy.nodes[node_id]
            queue.extend(node.children)
        
        return result
    
    def retrieve(self, query: str, 
                 threshold: float = 0.5,
                 top_k: int = 20,
                 strategy: str = 'moderate') -> List[RetrievalResult]:
        """
        Main retrieval function.
        
        Args:
            query: Query string
            threshold: Wu-Palmer similarity threshold (override strategy)
            top_k: Maximum results per query entity
            strategy: Threshold strategy ('strict', 'moderate', 'loose', 'exploratory')
        
        Returns:
            Merged and deduplicated results from all query entities
        """
        # Override threshold if strategy is provided
        if strategy != 'moderate':
            from wu_palmer import AdaptiveThresholdStrategy
            threshold = AdaptiveThresholdStrategy.get_threshold(strategy)
        
        logger.info(f"Retrieving for query: '{query}' (threshold={threshold}, strategy={strategy})")
        
        # Extract query entities
        query_entities = self.extract_query_entities(query)
        
        if not query_entities:
            logger.warning("No entities found in query, falling back to full-text search")
            return self._fallback_keyword_search(query, top_k)
        
        # Retrieve from each query entity
        all_results = {}  # node_id → best result
        
        for qe_id in query_entities:
            results = self.search_from_node(qe_id, threshold, top_k)
            
            for result in results:
                if result.node_id not in all_results:
                    all_results[result.node_id] = result
                else:
                    # Keep result with higher similarity
                    if result.similarity > all_results[result.node_id].similarity:
                        all_results[result.node_id] = result
        
        # Convert to list and sort
        merged_results = list(all_results.values())
        merged_results.sort(key=lambda r: r.similarity, reverse=True)
        
        logger.info(f"Retrieved {len(merged_results)} unique results")
        
        return merged_results[:top_k * len(query_entities)]
    
    def _fallback_keyword_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Fallback to simple keyword search if no entities found."""
        query_lower = query.lower()
        results = []
        
        for node_id, node in self.taxonomy.nodes.items():
            # Skip root and virtual nodes
            if node.node_type in ['VirtualRoot', 'Category']:
                continue
            
            # Simple keyword matching in name and description
            score = 0.0
            if query_lower in node.name.lower():
                score += 0.5
            if node.description and query_lower in node.description.lower():
                score += 0.3
            
            if score > 0:
                result = RetrievalResult(
                    node_id=node_id,
                    name=node.name,
                    description=node.description,
                    node_type=node.node_type,
                    modality=node.modality,
                    depth=node.depth,
                    similarity=score,
                    path_length=0,
                    lca_node=node_id
                )
                results.append(result)
        
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]
    
    def assemble_hierarchical_context(self, results: List[RetrievalResult],
                                      include_ancestors: bool = True) -> str:
        """
        Assemble context from retrieval results in hierarchical order.
        
        Order:
        1. Deep (specific): Retrieved entities with high similarity
        2. Mid (category): Parent categories for context
        3. Broad (general): Higher-level summaries
        
        Args:
            results: Retrieval results
            include_ancestors: Whether to include ancestor context
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Group by depth (deep to shallow)
        depth_groups = defaultdict(list)
        for result in results:
            depth_groups[result.depth].append(result)
        
        # Sort depths descending (deepest first = most specific)
        for depth in sorted(depth_groups.keys(), reverse=True):
            group = depth_groups[depth]
            context_parts.append(f"\n--- Depth {depth} (Specific) ---\n")
            
            for result in group:
                context_parts.append(
                    f"[{result.name}] ({result.node_type}, {result.modality}, "
                    f"similarity={result.similarity:.3f})\n"
                    f"{result.description}\n"
                )
        
        # Add ancestor context if requested
        if include_ancestors and results:
            context_parts.append(f"\n--- Hierarchical Context ---\n")
            
            # Get unique ancestors
            ancestors_added = set()
            for result in results[:5]:  # Top 5 results
                current = result.node_id
                path = []
                
                while current is not None and current not in ancestors_added:
                    node = self.taxonomy.nodes[current]
                    if node.node_type not in ['VirtualRoot']:
                        path.append(node.name)
                        ancestors_added.add(current)
                    current = node.parent_id
                
                if path:
                    context_parts.append(f"Path: {' → '.join(reversed(path))}\n")
        
        return ''.join(context_parts)
    
    def explain_retrieval(self, query: str, result: RetrievalResult) -> str:
        """
        Explain why a particular result was retrieved.
        
        This provides interpretability - showing the LCA path and similarity reasoning.
        """
        query_entities = self.extract_query_entities(query)
        
        if not query_entities:
            return "Result found via keyword matching"
        
        # Find which query entity led to this result
        best_qe = None
        best_sim = 0.0
        
        for qe_id in query_entities:
            sim = self.wp.similarity(qe_id, result.node_id)
            if sim > best_sim:
                best_sim = sim
                best_qe = qe_id
        
        if best_qe is None:
            return "Result found via expanded search"
        
        query_node = self.taxonomy.nodes[best_qe]
        lca_node_id = self.lca.lca(best_qe, result.node_id)
        lca_node = self.taxonomy.nodes[lca_node_id]
        
        explanation = f"""
Retrieval Explanation:
  Query Entity: {query_node.name} (depth={query_node.depth})
  Retrieved: {result.name} (depth={result.depth})
  Wu-Palmer Similarity: {result.similarity:.3f}
  Path Length: {result.path_length} edges
  Lowest Common Ancestor: {lca_node.name} (depth={lca_node.depth})
  
  Reasoning: Both entities share the common ancestor '{lca_node.name}',
  indicating they are semantically related within the same taxonomic branch.
  The Wu-Palmer similarity of {result.similarity:.3f} suggests a 
  {'strong' if result.similarity > 0.7 else 'moderate' if result.similarity > 0.5 else 'weak'} 
  semantic relationship.
"""
        return explanation
