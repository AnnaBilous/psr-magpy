import sys
import os

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) 