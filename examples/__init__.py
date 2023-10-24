import os
import sys
from pathlib import Path

path = Path(os.path.dirname(os.path.realpath(__file__))).parent
print(str(path))
sys.path.append(str(path))
