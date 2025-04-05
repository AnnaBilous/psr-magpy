#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Load the package metadata from pyproject.toml
# This is a simple setup.py that delegates most configuration to pyproject.toml

setup(
    name="psr_magpy",
    version="0.1.0",
    description="Magnetic field line tracing package for pulsar magnetosphere modeling",
    author="Anna Bilous",
    author_email="hanna.bilous@gmail.com",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "astropy>=5.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "psr-magpy-setup=psr_magpy.cli.setup_environment:main",
            "psr-magpy-setup-env=psr_magpy.cli.setup_environment:main",
        ],
    },
    data_files=[("share/psr_magpy", ["README.md"])],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
) 