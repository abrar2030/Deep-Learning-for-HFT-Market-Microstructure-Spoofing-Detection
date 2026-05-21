"""
pytest configuration: add the code/ directory to sys.path so that
bare `from models.` and `from utils.` imports resolve correctly
when running `pytest` from the project root or the code/ directory.
"""

import sys
from pathlib import Path

# code/ is the parent of tests/
code_root = Path(__file__).parent.parent
if str(code_root) not in sys.path:
    sys.path.insert(0, str(code_root))
