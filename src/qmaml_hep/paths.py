"""Where results and datasets live. Both default to directories under the current working directory (run
commands from the repository root) and can be redirected with environment variables."""

import os

RESULTS = os.environ.get("QMAML_RESULTS", "results")
DATA = os.environ.get("QMAML_DATA", "data")
