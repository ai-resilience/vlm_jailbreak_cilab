# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# check if python version is above 3.10
# Setup local transformers path before any other imports
import sys
import os
from pathlib import Path

# Get the directory where this __init__.py is located
_DEEPSEEK_VL2_DIR = Path(__file__).parent
_TRANSFORMERS_PATH = _DEEPSEEK_VL2_DIR / "transformers_4_38_2"

# Check for transformers_4_38_2/src/transformers/__init__.py structure
if _TRANSFORMERS_PATH.exists() and (_TRANSFORMERS_PATH / "src" / "transformers" / "__init__.py").exists():
    _local_transformers_path = str(_TRANSFORMERS_PATH / "src")
    if _local_transformers_path not in sys.path:
        sys.path.insert(0, _local_transformers_path)
        env_path = os.environ.get("PYTHONPATH", "")
        if _local_transformers_path not in env_path:
            os.environ["PYTHONPATH"] = _local_transformers_path + os.pathsep + env_path


# check if python version is above 3.10
if sys.version_info >= (3, 10):
    print("Python version is above 3.10, patching the collections module.")
    # Monkey patch collections
    import collections
    import collections.abc

    for type_name in collections.abc.__all__:
        setattr(collections, type_name, getattr(collections.abc, type_name))
