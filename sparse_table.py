"""
Sparse Table Implementation for O(1) LCA Queries
Using Euler Tour + Range Minimum Query (RMQ)

This module provides:
- Euler tour construction from taxonomy tree
- Sparse table for O(1) LCA queries
- O(n log n) preprocessing, O(1) query time
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np


class SparseTable:
    """
    Sparse Table for Range Minimum Query (RMQ) to support O(1) LCA.
    
    Preprocessing: O(n log n)
    Query: O(1)
    """
    
    def __init__(self, array: List[int], depths: List[int]):
        """
        Initialize sparse table for RMQ.
        
        Args:
            array: Euler tour array (node indices)
            depths: Depth of each node in euler tour
        """
        self.array = array
        self.depths = depths
        n = len(array)
        
        if n == 0:
            self.table = []
            self.log = []
            return
        
        # Precompute log values
        self.log = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log[i] = self.log[i // 2] + 1
        
        # Build sparse table
        max_log = self.log[n] + 1
        self.table = [[0] * n for _ in range(max_log)]
        
        # Initialize first row (intervals of length 1)
        for i in range(n):
            self.table[0][i] = i
        
        # Fill table using dynamic programming
        j = 1
        while (1 << j) <= n:
            i = 0
            while (i + (1 << j) - 1) < n:
                left = self.table[j - 1][i]
                right = self.table[j - 1][i + (1 << (j - 1))]
                
                # Choose index with minimum depth
                if self.depths[left] < self.depths[right]:
                    self.table[j][i] = left
                else:
                    self.table[j][i] = right
                i += 1
            j += 1
    
    def query(self, l: int, r: int) -> int:
        """
        Query minimum depth node in range [l, r].
        
        Args:
            l: Left index (inclusive)
            r: Right index (inclusive)
        
        Returns:
            Index in euler tour with minimum depth in range [l, r]
        """
        if l > r:
            l, r = r, l
        
        if l >= len(self.array) or r >= len(self.array):
            return 0
        
        # Find k such that 2^k <= (r - l + 1)
        length = r - l + 1
        k = self.log[length]
        
        # Query overlapping intervals
        left = self.table[k][l]
        right = self.table[k][r - (1 << k) + 1]
        
        # Return index with minimum depth
        if self.depths[left] < self.depths[right]:
            return left
        else:
            return right


class EulerTourLCA:
    """
    Lowest Common Ancestor using Euler Tour + Sparse Table.
    
    Features:
    - O(n log n) preprocessing
    - O(1) LCA queries
    - Deterministic and reproducible
    """
    
    def __init__(self):
        self.euler_tour: List[int] = []
        self.depths: List[int] = []
        self.first_occurrence: Dict[int, int] = {}
        self.node_depths: Dict[int, int] = {}
        self.sparse_table: Optional[SparseTable] = None
    
    def build(self, tree: Dict[int, List[int]], root: int, node_depths: Dict[int, int]):
        """
        Build Euler tour and sparse table for LCA queries.
        
        Args:
            tree: Adjacency list {parent_id: [child_ids]}
            root: Root node ID
            node_depths: Precomputed depths {node_id: depth}
        """
        self.euler_tour = []
        self.depths = []
        self.first_occurrence = {}
        self.node_depths = node_depths
        
        # Perform DFS to build Euler tour
        self._dfs(tree, root, 0)
        
        # Build sparse table on depths
        self.sparse_table = SparseTable(self.euler_tour, self.depths)
    
    def _dfs(self, tree: Dict[int, List[int]], node: int, depth: int):
        """
        DFS to construct Euler tour.
        
        Args:
            tree: Adjacency list
            node: Current node
            depth: Current depth
        """
        # First occurrence of this node
        if node not in self.first_occurrence:
            self.first_occurrence[node] = len(self.euler_tour)
        
        self.euler_tour.append(node)
        self.depths.append(depth)
        
        # Visit children
        if node in tree:
            for child in tree[node]:
                self._dfs(tree, child, depth + 1)
                # Add parent again after visiting child
                self.euler_tour.append(node)
                self.depths.append(depth)
    
    def lca(self, u: int, v: int) -> int:
        """
        Find Lowest Common Ancestor of nodes u and v in O(1) time.
        
        Args:
            u: First node ID
            v: Second node ID
        
        Returns:
            LCA node ID
        """
        if u not in self.first_occurrence or v not in self.first_occurrence:
            return -1
        
        l = self.first_occurrence[u]
        r = self.first_occurrence[v]
        
        if l > r:
            l, r = r, l
        
        # Query sparse table for minimum depth in range
        min_idx = self.sparse_table.query(l, r)
        return self.euler_tour[min_idx]
    
    def get_depth(self, node: int) -> int:
        """Get depth of a node."""
        return self.node_depths.get(node, 0)
    
    def save(self, filepath: str):
        """Save Euler tour and sparse table to file."""
        import pickle
        data = {
            'euler_tour': self.euler_tour,
            'depths': self.depths,
            'first_occurrence': self.first_occurrence,
            'node_depths': self.node_depths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load Euler tour and sparse table from file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.euler_tour = data['euler_tour']
        self.depths = data['depths']
        self.first_occurrence = data['first_occurrence']
        self.node_depths = data['node_depths']
        
        # Rebuild sparse table
        self.sparse_table = SparseTable(self.euler_tour, self.depths)
    
    def stats(self) -> Dict:
        """Get statistics about the LCA structure."""
        return {
            'num_nodes': len(self.first_occurrence),
            'euler_tour_length': len(self.euler_tour),
            'max_depth': max(self.node_depths.values()) if self.node_depths else 0,
            'avg_depth': sum(self.node_depths.values()) / len(self.node_depths) if self.node_depths else 0
        }


def test_lca():
    """Test LCA implementation with a simple tree."""
    #       1
    #      / \
    #     2   3
    #    / \   \
    #   4   5   6
    
    tree = {
        1: [2, 3],
        2: [4, 5],
        3: [6]
    }
    
    node_depths = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
    
    lca_solver = EulerTourLCA()
    lca_solver.build(tree, root=1, node_depths=node_depths)
    
    # Test cases
    test_cases = [
        (4, 5, 2),  # LCA(4, 5) = 2
        (4, 6, 1),  # LCA(4, 6) = 1
        (2, 3, 1),  # LCA(2, 3) = 1
        (4, 2, 2),  # LCA(4, 2) = 2 (ancestor)
        (5, 5, 5),  # LCA(5, 5) = 5 (same node)
    ]
    
    print("Testing LCA Implementation:")
    print("-" * 50)
    for u, v, expected in test_cases:
        result = lca_solver.lca(u, v)
        status = "✓" if result == expected else "✗"
        print(f"{status} LCA({u}, {v}) = {result} (expected {expected})")
    
    print("\nStatistics:")
    print(lca_solver.stats())


if __name__ == "__main__":
    test_lca()
