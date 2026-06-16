#!/usr/bin/env python3
"""
MetricEPI - Distance-Aware Deep Learning for Enhancer-Promoter Interaction Prediction
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="metricepi",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Distance-aware deep learning for enhancer-promoter interaction prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/MetricEPI",
    license="MIT",
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
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords=[
        "deep learning",
        "genomics",
        "enhancer-promoter interaction",
        "transformer",
        "attention mechanism",
        "bioinformatics",
    ],
)
