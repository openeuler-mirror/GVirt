import sys
from pathlib import Path

# The harness modules live one directory up (.gitcode/); keeping this in
# conftest lets the tests import them like normal modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
