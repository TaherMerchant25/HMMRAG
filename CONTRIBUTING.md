# Contributing to LeanRAG-MM

Thank you for your interest in contributing to LeanRAG-MM! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/leanrag-mm.git
   cd leanrag-mm
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install in editable mode with dev dependencies
   ```

## Testing

Run all tests before submitting a pull request:

```bash
bash test_all.sh
```

Or run individual tests:
```bash
python sparse_table.py
python wu_palmer.py
python multimodal_extractor.py
python pipeline.py --mode demo
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and under 50 lines when possible
- Add type hints where appropriate

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### PR Guidelines

- Clearly describe what your PR does and why
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed
- Keep PRs focused on a single feature/fix

## Areas for Contribution

### High Priority
- [ ] Enhanced multimodal extractors (audio, video)
- [ ] Additional evaluation metrics
- [ ] Visualization tools for taxonomy trees
- [ ] Performance benchmarks against other RAG systems

### Medium Priority
- [ ] More sophisticated entity resolution
- [ ] Support for incremental taxonomy updates
- [ ] Query optimization techniques
- [ ] Integration with popular LLM frameworks

### Documentation
- [ ] More usage examples
- [ ] Tutorial notebooks
- [ ] API documentation
- [ ] Architecture diagrams

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the architecture
- General discussion

## Code of Conduct

Be respectful, constructive, and professional in all interactions.
