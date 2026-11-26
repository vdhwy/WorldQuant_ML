import os
import sys

# Make the repo root importable so tests can `import lib` and `import core.*`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
