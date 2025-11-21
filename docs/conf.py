# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# Add the project source directory to the path
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'psr-magpy'
copyright = f'{datetime.now().year}, PSR-MagPy Team'
author = 'PSR-MagPy Team'
version = '0.1.0'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'nbsphinx',
    'matplotlib.sphinxext.plot_directive',
    # 'sphinx_autodoc_typehints',  # Disabled due to type hint compatibility issues
]

# Napoleon settings for NumPy docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'  # Use short format for type hints
autodoc_default_options = {
    'members': True,
    'undoc-members': False,  # Don't show undocumented members
    'inherited-members': False,  # Don't show inherited members
    'show-inheritance': True,
    'special-members': '__init__',
}
autosummary_generate = True  # Generate autosummary files
autosummary_imported_members = False

# Type hints settings
typehints_fully_qualified = False
always_document_param_types = False
typehints_document_rtype = True

# Intersphinx mapping (optional - will fail gracefully if servers are unavailable)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'astropy': ('https://docs.astropy.org/en/stable', None),
}
intersphinx_disabled_reftypes = ['std:label']  # Disable some ref types to reduce errors

# HTML output settings
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
# Logo and favicon are optional - only set if files exist
import os
if os.path.exists(os.path.join(html_static_path[0], 'logo.png')):
    html_logo = '_static/logo.png'
if os.path.exists(os.path.join(html_static_path[0], 'favicon.ico')):
    html_favicon = '_static/favicon.ico'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'navigation_depth': 4,
    'collapse_navigation': False,
}

# LaTeX settings for PDF output
latex_elements = {
    'preamble': r'''
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{physics}
    ''',
}

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Default role
default_role = 'py:obj'

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# Enable markdown support
extensions.append('myst_parser')
source_suffix = ['.rst', '.md'] 