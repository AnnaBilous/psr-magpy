#!/usr/bin/env python
"""
Documentation build script for psr-magpy.

This script automates the process of building documentation,
including API reference generation and example notebooks.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and print output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout

def clean_build_dir(docs_dir):
    """Clean the build directory."""
    build_dir = os.path.join(docs_dir, '_build')
    if os.path.exists(build_dir):
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)

def generate_api_docs(project_dir, docs_dir):
    """Generate API documentation using sphinx-apidoc."""
    src_dir = os.path.join(project_dir, 'src', 'psr_magpy')
    api_dir = os.path.join(docs_dir, 'api')
    
    # Ensure API directory exists
    os.makedirs(api_dir, exist_ok=True)
    
    # Run sphinx-apidoc
    cmd = [
        'sphinx-apidoc',
        '-f',  # Force overwrite
        '-e',  # Put each module on its own page
        '-o', api_dir,  # Output directory
        src_dir,  # Source directory
    ]
    run_command(cmd)
    
    print(f"API documentation generated in {api_dir}")

def copy_notebooks(project_dir, docs_dir):
    """Copy example notebooks to the docs directory."""
    notebooks_src = os.path.join(project_dir, 'demos')
    notebooks_dst = os.path.join(docs_dir, 'examples', 'notebooks')
    
    # Ensure destination directory exists
    os.makedirs(notebooks_dst, exist_ok=True)
    
    # Copy notebooks
    for nb_file in Path(notebooks_src).glob('*.ipynb'):
        if not nb_file.name.startswith('.'):  # Skip hidden files
            shutil.copy2(nb_file, notebooks_dst)
    
    print(f"Example notebooks copied to {notebooks_dst}")

def build_docs(docs_dir, format='html'):
    """Build the documentation using Sphinx."""
    build_dir = os.path.join(docs_dir, '_build', format)
    
    # Run sphinx-build
    cmd = [
        'sphinx-build',
        '-b', format,  # Build format
        '-d', os.path.join(docs_dir, '_build', 'doctrees'),  # Doctrees directory
        docs_dir,  # Source directory
        build_dir,  # Output directory
    ]
    run_command(cmd)
    
    print(f"{format.upper()} documentation built in {build_dir}")
    
    # If HTML, print the index.html path
    if format.lower() == 'html':
        index_path = os.path.join(build_dir, 'index.html')
        print(f"You can view the documentation at: file://{os.path.abspath(index_path)}")

def main():
    """Main function to build documentation."""
    parser = argparse.ArgumentParser(description='Build PSR-MagPy documentation')
    parser.add_argument('--clean', action='store_true', help='Clean build directory before building')
    parser.add_argument('--format', choices=['html', 'pdf', 'all'], default='html', 
                        help='Documentation format to build (default: html)')
    parser.add_argument('--no-api', action='store_true', help='Skip API documentation generation')
    parser.add_argument('--no-notebooks', action='store_true', help='Skip notebook copying')
    
    args = parser.parse_args()
    
    # Get project and docs directories
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    docs_dir = os.path.join(project_dir, 'docs')
    
    print(f"Project directory: {project_dir}")
    print(f"Documentation directory: {docs_dir}")
    
    # Clean build directory if requested
    if args.clean:
        clean_build_dir(docs_dir)
    
    # Generate API documentation
    if not args.no_api:
        generate_api_docs(project_dir, docs_dir)
    
    # Copy notebooks
    if not args.no_notebooks:
        copy_notebooks(project_dir, docs_dir)
    
    # Build documentation
    if args.format == 'all':
        build_docs(docs_dir, 'html')
        build_docs(docs_dir, 'pdf')
    else:
        build_docs(docs_dir, args.format)
    
    print("Documentation build complete!")

if __name__ == "__main__":
    main() 