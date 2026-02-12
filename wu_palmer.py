"""
Wu-Palmer Similarity using O(1) LCA Queries

Wu-Palmer similarity is a semantic distance metric based on the depth
of the Lowest Common Ancestor (LCA) in a taxonomy tree.

Formula:
    wu_palmer(u, v) = 2 * depth(lca(u, v)) / (depth(u) + depth(v))

Properties:
- Range: [0, 1]
- 1.0 = identical nodes
- 0.0 = no common ancestor (unrelated)
- Higher values = more semantically similar
"""

from typing import Dict, List, Tuple, Optional
import math


class WuPalmerSimilarity:
    """
    Wu-Palmer similarity computation using precomputed LCA.
    
    Time Complexity:
    - Preprocessing: O(n log n) for LCA structure
    - Query: O(1) per pair
    
    This is a massive improvement over embedding-based similarity:
    - Original: O(d) where d=1536 (embedding dimension)
    - Ours: O(1)
    """
    
    def __init__(self, lca_solver):
        """
        Initialize Wu-Palmer similarity calculator.
        
        Args:
            lca_solver: EulerTourLCA instance with precomputed LCA structure
        """
        self.lca_solver = lca_solver
    
    def similarity(self, u: int, v: int) -> float:
        """
        Compute Wu-Palmer similarity between two nodes.
        
        Args:
            u: First node ID
            v: Second node ID
        
        Returns:
            Similarity score in [0, 1]
        """
        # Handle same node
        if u == v:
            return 1.0
        
        # Get depths
        depth_u = self.lca_solver.get_depth(u)
        depth_v = self.lca_solver.get_depth(v)
        
        # Handle root or invalid nodes
        if depth_u == 0 or depth_v == 0:
            return 0.0
        
        # Find LCA
        lca_node = self.lca_solver.lca(u, v)
        if lca_node == -1:
            return 0.0
        
        depth_lca = self.lca_solver.get_depth(lca_node)
        
        # Wu-Palmer formula
        similarity = (2.0 * depth_lca) / (depth_u + depth_v)
        
        return similarity
    
    def batch_similarity(self, query_node: int, candidate_nodes: List[int]) -> List[Tuple[int, float]]:
        """
        Compute Wu-Palmer similarity between query node and multiple candidates.
        
        Args:
            query_node: Query node ID
            candidate_nodes: List of candidate node IDs
        
        Returns:
            List of (node_id, similarity) tuples, sorted by similarity (descending)
        """
        results = []
        for candidate in candidate_nodes:
            sim = self.similarity(query_node, candidate)
            results.append((candidate, sim))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def is_similar(self, u: int, v: int, threshold: float = 0.5) -> bool:
        """
        Check if two nodes are similar above a threshold.
        
        Args:
            u: First node ID
            v: Second node ID
            threshold: Similarity threshold (default 0.5)
        
        Returns:
            True if similarity >= threshold
        """
        return self.similarity(u, v) >= threshold
    
    def path_length(self, u: int, v: int) -> int:
        """
        Compute path length between two nodes through their LCA.
        
        Path length = depth(u) + depth(v) - 2 * depth(lca(u, v))
        
        Args:
            u: First node ID
            v: Second node ID
        
        Returns:
            Number of edges in path from u to v
        """
        if u == v:
            return 0
        
        lca_node = self.lca_solver.lca(u, v)
        if lca_node == -1:
            return float('inf')
        
        depth_u = self.lca_solver.get_depth(u)
        depth_v = self.lca_solver.get_depth(v)
        depth_lca = self.lca_solver.get_depth(lca_node)
        
        return depth_u + depth_v - 2 * depth_lca


class AdaptiveThresholdStrategy:
    """
    Adaptive threshold strategy for Wu-Palmer similarity.
    
    Different retrieval scenarios require different thresholds:
    - Strict (0.8+): Only very closely related entities (siblings, direct ancestors)
    - Moderate (0.5-0.8): Same category/domain
    - Loose (0.3-0.5): Broader relationships
    """
    
    THRESHOLDS = {
        'strict': 0.8,      # Same subtree, very close
        'moderate': 0.5,    # Same category
        'loose': 0.3,       # Related domain
        'exploratory': 0.1  # Any connection
    }
    
    @staticmethod
    def get_threshold(strategy: str = 'moderate') -> float:
        """Get threshold for given strategy."""
        return AdaptiveThresholdStrategy.THRESHOLDS.get(strategy, 0.5)
    
    @staticmethod
    def classify_relationship(similarity: float) -> str:
        """
        Classify relationship based on Wu-Palmer similarity.
        
        Args:
            similarity: Wu-Palmer similarity score
        
        Returns:
            Relationship category
        """
        if similarity >= 0.9:
            return "Very Close (siblings/direct ancestor)"
        elif similarity >= 0.7:
            return "Close (same subtree)"
        elif similarity >= 0.5:
            return "Related (same category)"
        elif similarity >= 0.3:
            return "Loosely Related (broader domain)"
        else:
            return "Distant (weak connection)"


def test_wu_palmer():
    """Test Wu-Palmer similarity with example tree."""
    from sparse_table import EulerTourLCA
    
    #       ROOT (0)
    #         |
    #      Entity (1)
    #      /    \
    #  Person  Concept
    #   (2)      (3)
    #   / \       |
    # Einstein Bohr  Physics
    #   (4)   (5)    (6)
    
    tree = {
        0: [1],
        1: [2, 3],
        2: [4, 5],
        3: [6]
    }
    
    node_depths = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3}
    node_names = {
        0: "ROOT",
        1: "Entity",
        2: "Person",
        3: "Concept",
        4: "Einstein",
        5: "Bohr",
        6: "Physics"
    }
    
    # Build LCA structure
    lca_solver = EulerTourLCA()
    lca_solver.build(tree, root=0, node_depths=node_depths)
    
    # Initialize Wu-Palmer calculator
    wp = WuPalmerSimilarity(lca_solver)
    
    # Test cases
    test_cases = [
        (4, 5, "Einstein vs Bohr (siblings)"),
        (4, 6, "Einstein vs Physics (different branches)"),
        (4, 2, "Einstein vs Person (child-parent)"),
        (5, 6, "Bohr vs Physics (cousins)"),
        (4, 4, "Einstein vs Einstein (same)"),
    ]
    
    print("Testing Wu-Palmer Similarity:")
    print("=" * 70)
    for u, v, description in test_cases:
        sim = wp.similarity(u, v)
        path_len = wp.path_length(u, v)
        relationship = AdaptiveThresholdStrategy.classify_relationship(sim)
        
        print(f"\n{description}")
        print(f"  {node_names[u]} <-> {node_names[v]}")
        print(f"  Wu-Palmer Similarity: {sim:.3f}")
        print(f"  Path Length: {path_len}")
        print(f"  Relationship: {relationship}")
    
    print("\n" + "=" * 70)
    print("Threshold Strategies:")
    for strategy, threshold in AdaptiveThresholdStrategy.THRESHOLDS.items():
        print(f"  {strategy:15s}: {threshold:.2f}")


if __name__ == "__main__":
    test_wu_palmer()
