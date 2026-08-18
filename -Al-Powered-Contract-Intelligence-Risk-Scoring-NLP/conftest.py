"""
Root conftest.py — ensures the project root is on sys.path for all tests.
"""
import sys
from pathlib import Path

# Insert project root so that `import ingestion`, `import ner`, etc. work
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
