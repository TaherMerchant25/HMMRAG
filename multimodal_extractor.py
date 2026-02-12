"""
Multimodal Extractor - Handle Text, Images, Tables, Audio

This module extends LeanRAG to multimodal data by:
1. Extracting entities from different modalities
2. Creating cross-modal links
3. Placing multimodal entities in unified taxonomy
4. Enabling cross-modal retrieval through shared hierarchy

Supported Modalities:
- Text: Traditional NER + relation extraction
- Images: Captions, OCR, object detection
- Tables: Schema + cell values + headers
- Audio: Transcripts + speaker diarization (future)
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class MultimodalEntity:
    """Entity extracted from any modality."""
    name: str
    entity_type: str
    modality: str  # 'text', 'image', 'table', 'audio'
    description: str
    source_id: str
    metadata: Dict  # Modality-specific metadata
    
    def to_triple_dict(self) -> Dict:
        """Convert to triple format for taxonomy builder."""
        return {
            'head': self.name,
            'head_type': self.entity_type,
            'head_description': self.description,
            'head_modality': self.modality,
            'relation': 'is_instance_of',
            'tail': self.entity_type,
            'tail_type': 'Concept',
            'tail_description': f"Category: {self.entity_type}",
            'tail_modality': 'text',
            'source_id': self.source_id
        }


class TextExtractor:
    """Extract entities from text documents."""
    
    def __init__(self):
        self.entity_patterns = {
            'Person': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Simple name pattern
            'Organization': r'\b[A-Z][A-Z]+\b',  # Acronyms
            'Concept': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'  # Title case phrases
        }
    
    def extract(self, text: str, source_id: str) -> List[MultimodalEntity]:
        """
        Extract entities from text.
        
        For production, this should use proper NER (spaCy, etc.)
        This is a simplified version for demonstration.
        """
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(0)
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                description = text[context_start:context_end]
                
                entity = MultimodalEntity(
                    name=name,
                    entity_type=entity_type,
                    modality='text',
                    description=description,
                    source_id=source_id,
                    metadata={'position': match.start()}
                )
                entities.append(entity)
        
        return entities


class ImageExtractor:
    """Extract entities from images via captions and object detection."""
    
    def __init__(self):
        pass
    
    def extract(self, image_path: str, caption: str = "", source_id: str = "") -> List[MultimodalEntity]:
        """
        Extract entities from image.
        
        In production, this would use:
        - Image captioning models (BLIP, CLIP)
        - Object detection (YOLO, Faster R-CNN)
        - OCR for text in images (Tesseract, PaddleOCR)
        
        For now, we extract from caption if provided.
        """
        entities = []
        
        if caption:
            # Extract from caption using text extraction
            text_extractor = TextExtractor()
            text_entities = text_extractor.extract(caption, source_id)
            
            # Convert to image modality
            for entity in text_entities:
                entity.modality = 'image'
                entity.metadata['image_path'] = image_path
                entity.metadata['caption'] = caption
                entities.append(entity)
        
        # Add the image itself as an entity
        image_name = os.path.basename(image_path)
        image_entity = MultimodalEntity(
            name=image_name,
            entity_type='Figure',
            modality='image',
            description=caption or f"Image: {image_name}",
            source_id=source_id,
            metadata={'path': image_path, 'caption': caption}
        )
        entities.append(image_entity)
        
        return entities


class TableExtractor:
    """Extract entities from structured tables."""
    
    def __init__(self):
        pass
    
    def extract(self, table_data: Dict, source_id: str = "") -> List[MultimodalEntity]:
        """
        Extract entities from table.
        
        Args:
            table_data: Dict with 'headers' and 'rows' keys
            source_id: Source identifier
        
        Returns:
            List of entities extracted from table
        """
        entities = []
        
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        # Extract header concepts
        for header in headers:
            entity = MultimodalEntity(
                name=header,
                entity_type='TableColumn',
                modality='table',
                description=f"Column: {header}",
                source_id=source_id,
                metadata={'header': header}
            )
            entities.append(entity)
        
        # Extract cell values (first column typically contains entities)
        if rows and len(rows[0]) > 0:
            for i, row in enumerate(rows):
                if row[0]:  # First column value
                    entity = MultimodalEntity(
                        name=str(row[0]),
                        entity_type='TableEntry',
                        modality='table',
                        description=f"Row {i+1}: {', '.join(map(str, row))}",
                        source_id=source_id,
                        metadata={'row': i, 'values': row}
                    )
                    entities.append(entity)
        
        return entities


class CrossModalLinker:
    """
    Link entities across modalities.
    
    Creates cross-modal relations like:
    - Text entity → Image entity (via co-occurrence)
    - Table entity → Text entity (via name matching)
    - Image entity → Text entity (via caption)
    """
    
    def __init__(self):
        pass
    
    def link(self, entities_by_modality: Dict[str, List[MultimodalEntity]]) -> List[Dict]:
        """
        Create cross-modal links between entities.
        
        Args:
            entities_by_modality: Dict mapping modality → list of entities
        
        Returns:
            List of cross-modal relation triples
        """
        cross_modal_triples = []
        
        text_entities = entities_by_modality.get('text', [])
        image_entities = entities_by_modality.get('image', [])
        table_entities = entities_by_modality.get('table', [])
        
        # Link text ↔ images (via name matching in captions)
        for text_e in text_entities:
            for image_e in image_entities:
                caption = image_e.metadata.get('caption', '')
                if text_e.name.lower() in caption.lower():
                    triple = {
                        'head': image_e.name,
                        'head_type': image_e.entity_type,
                        'head_description': image_e.description,
                        'head_modality': 'image',
                        'relation': 'illustrates',
                        'tail': text_e.name,
                        'tail_type': text_e.entity_type,
                        'tail_description': text_e.description,
                        'tail_modality': 'text',
                        'source_id': f"{image_e.source_id}↔{text_e.source_id}"
                    }
                    cross_modal_triples.append(triple)
        
        # Link text ↔ tables (via name matching)
        for text_e in text_entities:
            for table_e in table_entities:
                if text_e.name.lower() in table_e.description.lower():
                    triple = {
                        'head': table_e.name,
                        'head_type': table_e.entity_type,
                        'head_description': table_e.description,
                        'head_modality': 'table',
                        'relation': 'contains_data_about',
                        'tail': text_e.name,
                        'tail_type': text_e.entity_type,
                        'tail_description': text_e.description,
                        'tail_modality': 'text',
                        'source_id': f"{table_e.source_id}↔{text_e.source_id}"
                    }
                    cross_modal_triples.append(triple)
        
        logger.info(f"Created {len(cross_modal_triples)} cross-modal links")
        return cross_modal_triples


class MultimodalKGBuilder:
    """
    Build unified Knowledge Graph from multimodal data.
    
    Workflow:
    1. Extract entities from each modality
    2. Create cross-modal links
    3. Build taxonomy with modality-aware placement
    4. Enable cross-modal retrieval
    """
    
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.image_extractor = ImageExtractor()
        self.table_extractor = TableExtractor()
        self.cross_modal_linker = CrossModalLinker()
    
    def process_document(self, doc_data: Dict) -> Tuple[List[MultimodalEntity], List[Dict]]:
        """
        Process a multimodal document.
        
        Args:
            doc_data: Dict with keys:
                - 'text': Text content
                - 'images': List of {path, caption}
                - 'tables': List of {headers, rows}
                - 'source_id': Document identifier
        
        Returns:
            Tuple of (entities, cross_modal_triples)
        """
        source_id = doc_data.get('source_id', 'unknown')
        all_entities = []
        entities_by_modality = defaultdict(list)
        
        # Extract from text
        if 'text' in doc_data:
            text_entities = self.text_extractor.extract(doc_data['text'], source_id)
            all_entities.extend(text_entities)
            entities_by_modality['text'].extend(text_entities)
            logger.info(f"Extracted {len(text_entities)} text entities from {source_id}")
        
        # Extract from images
        if 'images' in doc_data:
            for img_data in doc_data['images']:
                img_path = img_data.get('path', '')
                caption = img_data.get('caption', '')
                img_entities = self.image_extractor.extract(img_path, caption, source_id)
                all_entities.extend(img_entities)
                entities_by_modality['image'].extend(img_entities)
            logger.info(f"Extracted {len(entities_by_modality['image'])} image entities from {source_id}")
        
        # Extract from tables
        if 'tables' in doc_data:
            for table_data in doc_data['tables']:
                table_entities = self.table_extractor.extract(table_data, source_id)
                all_entities.extend(table_entities)
                entities_by_modality['table'].extend(table_entities)
            logger.info(f"Extracted {len(entities_by_modality['table'])} table entities from {source_id}")
        
        # Create cross-modal links
        cross_modal_triples = self.cross_modal_linker.link(dict(entities_by_modality))
        
        return all_entities, cross_modal_triples
    
    def build_unified_taxonomy(self, documents: List[Dict], output_dir: str):
        """
        Build unified taxonomy from multiple multimodal documents.
        
        Args:
            documents: List of document dicts (see process_document)
            output_dir: Directory to save taxonomy and triples
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_entities = []
        all_cross_modal_triples = []
        
        # Process each document
        for doc_data in documents:
            entities, cross_modal_triples = self.process_document(doc_data)
            all_entities.extend(entities)
            all_cross_modal_triples.extend(cross_modal_triples)
        
        # Convert entities to triples
        entity_triples = [e.to_triple_dict() for e in all_entities]
        
        # Combine with cross-modal triples
        all_triples = entity_triples + all_cross_modal_triples
        
        # Save triples
        triples_path = os.path.join(output_dir, 'multimodal_triples.jsonl')
        with open(triples_path, 'w') as f:
            for triple in all_triples:
                f.write(json.dumps(triple) + '\n')
        
        logger.info(f"Saved {len(all_triples)} triples to {triples_path}")
        logger.info(f"  - Entity triples: {len(entity_triples)}")
        logger.info(f"  - Cross-modal triples: {len(all_cross_modal_triples)}")
        
        # Build taxonomy
        from taxonomy_builder import TaxonomyBuilder
        taxonomy = TaxonomyBuilder()
        taxonomy.build_from_triples(all_triples)
        
        # Save taxonomy
        taxonomy_path = os.path.join(output_dir, 'taxonomy.json')
        taxonomy.save(taxonomy_path)
        
        return taxonomy, all_triples


from collections import defaultdict


def demo_multimodal():
    """Demonstrate multimodal KG building."""
    
    # Example multimodal document
    documents = [
        {
            'source_id': 'einstein_paper',
            'text': 'Albert Einstein published groundbreaking work on the photoelectric effect. '
                   'This research contributed to quantum mechanics.',
            'images': [
                {
                    'path': 'figures/apparatus.png',
                    'caption': 'Photoelectric effect apparatus showing electron emission from metal surface'
                }
            ],
            'tables': [
                {
                    'headers': ['Experiment', 'Wavelength', 'Energy'],
                    'rows': [
                        ['Exp1', '400nm', '3.1eV'],
                        ['Exp2', '500nm', '2.5eV']
                    ]
                }
            ]
        }
    ]
    
    # Build unified taxonomy
    builder = MultimodalKGBuilder()
    taxonomy, triples = builder.build_unified_taxonomy(documents, '/tmp/multimodal_kg')
    
    print("\n" + "="*70)
    print("Multimodal Taxonomy Built Successfully!")
    print("="*70)
    print(f"Total entities: {len(taxonomy.nodes)}")
    print(f"Total triples: {len(triples)}")
    print("\nTaxonomy Structure:")
    taxonomy.print_tree(max_depth=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_multimodal()
