# This code is part of Qiskit.
#
# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

class _CudaQJob:
    def __init__(self, result_obj):
        self._result = result_obj

    def result(self):
        return self._result


def _require_cudaq():
    if cudaq is None:
        raise RuntimeError("CUDA-Q is not available. Run `pip install cudaq` and make sure it imports.")
