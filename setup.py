"""
Setup script for DocuStruct Multi-Model OCR Framework

This file provides backward compatibility for older build systems.
The primary configuration is in pyproject.toml.
"""

from setuptools import setup

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="manazir-ocr",
    use_scm_version=False,
    version="0.2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
