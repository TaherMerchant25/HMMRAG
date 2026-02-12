"""
Taxonomy Builder - Constructs Hierarchical KG from IS-A/TYPE-OF Relations

This module replaces the O(n²×d) embedding-based Louvain clustering with
an O(n log n) taxonomy-aware hierarchy using existing semantic relationships.

Key Features:
- Extracts taxonomic relations (IS-A, TYPE-OF, PART-OF, etc.)
- Builds deterministic tree structure
- Handles disconnected components with virtual root
- Assigns non-taxonomic entities to appropriate branches
- O(n log n) complexity vs O(n²×d) for Louvain
"""

import json
import os
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaxonomyNode:
    """Node in the taxonomy tree."""
    node_id: int
    name: str
    node_type: str  # "entity", "concept", "media", "data", etc.
    modality: str  # "text", "image", "table", "audio"
    depth: int
    parent_id: Optional[int]
    children: List[int]
    description: str = ""
    source_ids: List[str] = None
    
    def __post_init__(self):
        if self.source_ids is None:
            self.source_ids = []


class TaxonomyBuilder:
    """
    Build taxonomy tree from knowledge graph triples.
    
    Algorithm:
    1. Extract taxonomic relations (IS-A, TYPE-OF, etc.) → O(n)
    2. Build directed graph from parent→child edges → O(n)
    3. Detect and remove cycles (Kahn's algorithm) → O(n)
    4. Add virtual root for disconnected components → O(k) where k=#components
    5. Compute depths via DFS → O(n)
    6. Assign non-taxonomic entities → O(m) where m=#non-taxonomic entities
    
    Total: O(n + m) = O(n)
    """
    
    # Taxonomic relation patterns (case-insensitive)
    TAXONOMIC_RELATIONS = {
        'is', 'is_a', 'is_an', 'isa',
        'type_of', 'type', 'typeof',
        'instance_of', 'instanceof',
        'subclass_of', 'subclass',
        'part_of', 'partof',
        'belongs_to', 'belongsto',
        'include', 'includes',
        'category', 'category_of',
        'kind_of', 'kindof',
        'member_of', 'memberof'
    }
    
    def __init__(self, virtual_root_name: str = "ROOT"):
        self.virtual_root_name = virtual_root_name
        self.nodes: Dict[int, TaxonomyNode] = {}
        self.name_to_id: Dict[str, int] = {}
        self.adjacency: Dict[int, List[int]] = defaultdict(list)
        self.root_id: Optional[int] = None
        self.next_id = 0
    
    def _get_or_create_node(self, name: str, node_type: str = "Unknown", 
                           modality: str = "text", description: str = "") -> int:
        """Get existing node ID or create new node."""
        if name in self.name_to_id:
            return self.name_to_id[name]
        
        node_id = self.next_id
        self.next_id += 1
        
        node = TaxonomyNode(
            node_id=node_id,
            name=name,
            node_type=node_type,
            modality=modality,
            depth=-1,  # Will be computed later
            parent_id=None,
            children=[],
            description=description
        )
        
        self.nodes[node_id] = node
        self.name_to_id[name] = node_id
        return node_id
    
    def is_taxonomic_relation(self, relation: str) -> bool:
        """Check if a relation is taxonomic."""
        relation_normalized = relation.lower().replace(' ', '_').replace('-', '_')
        return relation_normalized in self.TAXONOMIC_RELATIONS
    
    def build_from_triples(self, triples: List[Dict]) -> int:
        """
        Build taxonomy tree from knowledge graph triples.
        
        Args:
            triples: List of dicts with keys: head, relation, tail, head_type, tail_type, etc.
        
        Returns:
            Root node ID
        """
        logger.info(f"Building taxonomy from {len(triples)} triples...")
        
        # Step 1: Extract taxonomic relations and create nodes
        taxonomic_edges = []
        entity_info = {}  # Store entity metadata
        
        for triple in triples:
            head = triple.get('head', triple.get('head_entity', ''))
            relation = triple.get('relation', '')
            tail = triple.get('tail', triple.get('tail_entity', ''))
            
            # Store entity metadata
            if head and head not in entity_info:
                entity_info[head] = {
                    'type': triple.get('head_type', 'Unknown'),
                    'desc': triple.get('head_description', ''),
                    'modality': triple.get('head_modality', 'text'),
                    'source_id': triple.get('source_id', '')
                }
            if tail and tail not in entity_info:
                entity_info[tail] = {
                    'type': triple.get('tail_type', 'Unknown'),
                    'desc': triple.get('tail_description', ''),
                    'modality': triple.get('tail_modality', 'text'),
                    'source_id': triple.get('source_id', '')
                }
            
            # Check if relation is taxonomic
            if self.is_taxonomic_relation(relation):
                # For IS-A relations: child IS-A parent, so edge is parent→child
                # But in triple format: child IS-A parent
                # So we need: tail (parent) → head (child)
                taxonomic_edges.append((tail, head))
        
        logger.info(f"Found {len(taxonomic_edges)} taxonomic relations")
        
        # Step 2: Create nodes for all entities in taxonomic relations
        for parent, child in taxonomic_edges:
            parent_info = entity_info.get(parent, {})
            child_info = entity_info.get(child, {})
            
            parent_id = self._get_or_create_node(
                parent, 
                parent_info.get('type', 'Concept'),
                parent_info.get('modality', 'text'),
                parent_info.get('desc', '')
            )
            child_id = self._get_or_create_node(
                child,
                child_info.get('type', 'Entity'),
                child_info.get('modality', 'text'),
                child_info.get('desc', '')
            )
            
            # Add edge: parent → child
            self.adjacency[parent_id].append(child_id)
            self.nodes[child_id].parent_id = parent_id
        
        # Step 3: Detect and handle cycles
        self._remove_cycles()
        
        # Step 4: Add virtual root for disconnected components
        self._add_virtual_root()
        
        # Step 5: Compute depths
        self._compute_depths()
        
        # Step 6: Assign non-taxonomic entities
        self._assign_orphan_entities(triples, entity_info)
        
        logger.info(f"Taxonomy built successfully:")
        logger.info(f"  - Total nodes: {len(self.nodes)}")
        logger.info(f"  - Root: {self.nodes[self.root_id].name}")
        logger.info(f"  - Max depth: {max(n.depth for n in self.nodes.values())}")
        
        return self.root_id
    
    def _remove_cycles(self):
        """Detect and remove cycles using DFS."""
        visited = set()
        rec_stack = set()
        cycles_removed = 0
        
        def has_cycle(node_id: int) -> bool:
            nonlocal cycles_removed
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for child_id in self.adjacency.get(node_id, []):
                if child_id not in visited:
                    if has_cycle(child_id):
                        return True
                elif child_id in rec_stack:
                    # Cycle detected! Remove this edge
                    self.adjacency[node_id].remove(child_id)
                    self.nodes[child_id].parent_id = None
                    cycles_removed += 1
                    logger.warning(f"Cycle detected: removed edge {node_id}→{child_id}")
                    return False
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in list(self.nodes.keys()):
            if node_id not in visited:
                has_cycle(node_id)
        
        if cycles_removed > 0:
            logger.info(f"Removed {cycles_removed} cyclic edges")
    
    def _add_virtual_root(self):
        """Add virtual root node and connect disconnected components."""
        # Find all nodes without parents (roots)
        roots = [nid for nid, node in self.nodes.items() if node.parent_id is None]
        
        if len(roots) == 0:
            # No nodes at all - create a single Unknown node
            logger.warning("No entities found in taxonomy. Creating minimal structure.")
            self.root_id = self._get_or_create_node(
                self.virtual_root_name,
                "VirtualRoot",
                "text",
                "Virtual root for knowledge graph"
            )
            self.nodes[self.root_id].depth = 0
            
            # Create an Unknown category
            unknown_id = self._get_or_create_node(
                "Unknown",
                "Category",
                "text",
                "Uncategorized entities"
            )
            self.adjacency[self.root_id].append(unknown_id)
            self.nodes[unknown_id].parent_id = self.root_id
            self.nodes[unknown_id].depth = 1
            self.nodes[self.root_id].children.append(unknown_id)
            
            logger.info(f"Created minimal taxonomy with root '{self.virtual_root_name}'")
            return
        
        if len(roots) == 1:
            # Single root, use it directly
            self.root_id = roots[0]
            self.nodes[self.root_id].depth = 0
            logger.info(f"Single root found: {self.nodes[self.root_id].name}")
        else:
            # Multiple roots, create virtual root
            self.root_id = self._get_or_create_node(
                self.virtual_root_name,
                "VirtualRoot",
                "text",
                "Virtual root connecting all top-level categories"
            )
            self.nodes[self.root_id].depth = 0
            
            for root in roots:
                self.adjacency[self.root_id].append(root)
                self.nodes[root].parent_id = self.root_id
            
            logger.info(f"Created virtual root '{self.virtual_root_name}' connecting {len(roots)} components")
    
    def _compute_depths(self):
        """Compute depth for all nodes via BFS from root."""
        queue = deque([self.root_id])
        self.nodes[self.root_id].depth = 0
        
        while queue:
            node_id = queue.popleft()
            current_depth = self.nodes[node_id].depth
            
            for child_id in self.adjacency.get(node_id, []):
                self.nodes[child_id].depth = current_depth + 1
                self.nodes[child_id].parent_id = node_id
                self.nodes[node_id].children.append(child_id)
                queue.append(child_id)
    
    def _assign_orphan_entities(self, triples: List[Dict], entity_info: Dict):
        """
        Assign entities not in taxonomic relations to appropriate branches.
        
        Strategy:
        1. If entity appears in any relation, try to find its type
        2. Find existing type node in taxonomy
        3. Attach entity to that type node
        4. If no type found, create "Unknown" category
        """
        # Get all entities mentioned in triples
        all_entities = set()
        for triple in triples:
            head = triple.get('head', triple.get('head_entity', ''))
            tail = triple.get('tail', triple.get('tail_entity', ''))
            if head:
                all_entities.add(head)
            if tail:
                all_entities.add(tail)
        
        # Find orphan entities (not in taxonomy yet)
        orphans = all_entities - set(self.name_to_id.keys())
        
        if not orphans:
            logger.info("No orphan entities to assign")
            return
        
        logger.info(f"Assigning {len(orphans)} orphan entities to taxonomy...")
        
        # Create or find "Unknown" category under root
        unknown_id = None
        for child_id in self.adjacency.get(self.root_id, []):
            if self.nodes[child_id].name.lower() == "unknown":
                unknown_id = child_id
                break
        
        if unknown_id is None:
            unknown_id = self._get_or_create_node("Unknown", "Category", "text", "Uncategorized entities")
            self.adjacency[self.root_id].append(unknown_id)
            self.nodes[unknown_id].parent_id = self.root_id
            self.nodes[unknown_id].depth = 1
            self.nodes[self.root_id].children.append(unknown_id)
        
        # Assign each orphan
        for entity in orphans:
            info = entity_info.get(entity, {})
            entity_type = info.get('type', 'Unknown')
            
            # Try to find type node in taxonomy
            parent_id = self.name_to_id.get(entity_type, unknown_id)
            
            # Create entity node
            entity_id = self._get_or_create_node(
                entity,
                entity_type,
                info.get('modality', 'text'),
                info.get('desc', '')
            )
            
            # Attach to parent
            self.adjacency[parent_id].append(entity_id)
            self.nodes[entity_id].parent_id = parent_id
            self.nodes[entity_id].depth = self.nodes[parent_id].depth + 1
            self.nodes[parent_id].children.append(entity_id)
        
        logger.info(f"Assigned {len(orphans)} orphan entities")
    
    def get_tree_adjacency(self) -> Dict[int, List[int]]:
        """Get adjacency list representation of tree."""
        return dict(self.adjacency)
    
    def get_node_depths(self) -> Dict[int, int]:
        """Get depth mapping for all nodes."""
        return {nid: node.depth for nid, node in self.nodes.items()}
    
    def save(self, filepath: str):
        """Save taxonomy to JSON file."""
        data = {
            'root_id': self.root_id,
            'nodes': {
                nid: {
                    'name': node.name,
                    'type': node.node_type,
                    'modality': node.modality,
                    'depth': node.depth,
                    'parent_id': node.parent_id,
                    'children': node.children,
                    'description': node.description,
                    'source_ids': node.source_ids
                }
                for nid, node in self.nodes.items()
            },
            'name_to_id': self.name_to_id
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Taxonomy saved to {filepath}")
    
    def load(self, filepath: str):
        """Load taxonomy from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.root_id = data['root_id']
        self.name_to_id = data['name_to_id']
        
        # Reconstruct nodes
        self.nodes = {}
        for nid_str, node_data in data['nodes'].items():
            nid = int(nid_str)
            node = TaxonomyNode(
                node_id=nid,
                name=node_data['name'],
                node_type=node_data['type'],
                modality=node_data['modality'],
                depth=node_data['depth'],
                parent_id=node_data['parent_id'],
                children=node_data['children'],
                description=node_data.get('description', ''),
                source_ids=node_data.get('source_ids', [])
            )
            self.nodes[nid] = node
        
        # Reconstruct adjacency
        self.adjacency = defaultdict(list)
        for nid, node in self.nodes.items():
            if node.children:
                self.adjacency[nid] = node.children
        
        self.next_id = max(self.nodes.keys()) + 1
        
        logger.info(f"Taxonomy loaded from {filepath}")
    
    def print_tree(self, node_id: Optional[int] = None, indent: int = 0, max_depth: int = 3):
        """Print tree structure (for debugging)."""
        if node_id is None:
            node_id = self.root_id
        
        node = self.nodes[node_id]
        if node.depth > max_depth:
            return
        
        prefix = "  " * indent
        print(f"{prefix}├─ {node.name} ({node.node_type}, depth={node.depth})")
        
        for child_id in node.children[:5]:  # Limit to first 5 children
            self.print_tree(child_id, indent + 1, max_depth)
        
        if len(node.children) > 5:
            print(f"{prefix}  └─ ... and {len(node.children) - 5} more")
