#!/bin/bash
# Quick test script for VATRAG 2.0

set -e

echo "=========================================="
echo "VATRAG 2.0 - Quick Test"
echo "=========================================="
echo ""

# Test 1: LCA Implementation
echo "[1/4] Testing LCA implementation..."
python3 sparse_table.py
echo ""

# Test 2: Wu-Palmer Similarity
echo "[2/4] Testing Wu-Palmer similarity..."
python3 wu_palmer.py
echo ""

# Test 3: Multimodal Extractor
echo "[3/4] Testing multimodal extraction..."
python3 multimodal_extractor.py
echo ""

# Test 4: Full Pipeline Demo
echo "[4/4] Running pipeline demo..."
python3 pipeline.py --mode demo
echo ""

echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Build from VATRAG data:"
echo "     python3 pipeline.py --mode build --input ../VATRAG/ckg_data/mix_chunk/new_triples_mix_chunk.jsonl"
echo ""
echo "  2. Query:"
echo "     python3 pipeline.py --mode query --query 'your question here'"
