"""
Setup script for LeanRAG-MM
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="leanrag-mm",
    version="2.0.0",
    author="LeanRAG-MM Team",
    description="LCA-Optimized Multimodal Knowledge Graph Retrieval with Wu-Palmer Semantic Distance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/leanrag-mm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "full": [
            "spacy>=3.5.0",
            "pillow>=9.0.0",
            "pandas>=1.5.0",
        ],
    },
)
