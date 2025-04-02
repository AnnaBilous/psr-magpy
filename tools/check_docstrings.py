#!/usr/bin/env python
"""
Docstring validation script for psr-magpy.

This script checks that docstrings follow the NumPy format and
contain all required sections.
"""

import os
import sys
import ast
import re
from pathlib import Path

def check_docstring(docstring, name):
    """
    Check if a docstring follows NumPy format.
    
    Parameters
    ----------
    docstring : str
        The docstring to check
    name : str
        Name of the function or class
        
    Returns
    -------
    list
        List of issues found, empty if docstring is valid
    """
    issues = []
    
    if not docstring:
        return [f"{name} has no docstring"]
    
    # Check for short summary
    if not docstring.strip().split('\n')[0].strip():
        issues.append(f"{name} is missing short summary line")
    
    # Check for Parameters section in functions with parameters
    if "def " in name and "(" in name and ")" in name:
        params = name.split('(')[1].split(')')[0].strip()
        if params and params != "self" and "Parameters" not in docstring:
            issues.append(f"{name} is missing Parameters section")
    
    # Check for Returns section in functions
    if "def " in name and "Returns" not in docstring and "yield" not in docstring.lower():
        issues.append(f"{name} may be missing Returns section")
    
    # Check for proper section formatting
    sections = ["Parameters", "Returns", "Raises", "See Also", "Notes", "Examples"]
    for section in sections:
        if section in docstring:
            # Check if section has proper underline
            pattern = f"{section}\\s*\\n\\s*-+\\s*\\n"
            if not re.search(pattern, docstring):
                issues.append(f"{name} has {section} section without proper formatting")
    
    return issues

def check_file(file_path):
    """
    Check all docstrings in a Python file.
    
    Parameters
    ----------
    file_path : str
        Path to the Python file
        
    Returns
    -------
    list
        List of docstring issues found
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                name = getattr(node, 'name', 'module') if not isinstance(node, ast.Module) else file_path
                if isinstance(node, ast.FunctionDef):
                    name = f"def {name}"
                elif isinstance(node, ast.ClassDef):
                    name = f"class {name}"
                
                node_issues = check_docstring(docstring, name)
                for issue in node_issues:
                    issues.append(f"{file_path}: {issue}")
    
    except SyntaxError as e:
        issues.append(f"{file_path}: Syntax error: {str(e)}")
    
    return issues

def main():
    """Check docstrings in all Python files in the project."""
    src_dir = Path("psr-magpy/src/psr_magpy")
    
    all_issues = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                issues = check_file(file_path)
                all_issues.extend(issues)
    
    if all_issues:
        for issue in all_issues:
            print(issue)
        sys.exit(1)
    else:
        print("All docstrings follow NumPy format!")
        sys.exit(0)

if __name__ == "__main__":
    main() 