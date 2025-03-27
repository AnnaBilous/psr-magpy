"""Setup script for psr-magpy package."""

from setuptools import setup, find_packages

setup(
    name="psr_magpy",
    version="0.1.0",
    description="Magnetic field line tracing package for pulsar magnetosphere modeling",
    author="Anna Bilous",  # Replace with your name
    author_email="hanna.bilous@gmail.com",  # Replace with your email
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
    python_requires=">=3.8",
) 