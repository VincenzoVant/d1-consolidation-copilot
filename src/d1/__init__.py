"""D1 Consolidation Copilot: account mapping and intercompany matching on synthetic ledgers."""

import os as _os

# One OpenMP thread, set before any library loads OpenMP.
#
# XGBoost and PyTorch each ship their own copy of the OpenMP library (libomp). On macOS, with both
# loaded in one process, running Laya after importing XGBoost, or training XGBoost after running
# Laya, ends in a segmentation fault: no Python error, the process just dies. With one OpenMP
# thread the two copies never collide. It costs little here: bge-m3 and Laya run on the GPU
# (Metal), which doesn't use OpenMP, and an XGBoost training run took 49 s instead of 42 s with
# byte-identical results. `setdefault`: a value set on purpose before starting Python is kept.
_os.environ.setdefault("OMP_NUM_THREADS", "1")
