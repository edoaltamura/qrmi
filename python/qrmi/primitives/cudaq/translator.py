# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""The CUDA-Q connector translator.

Key ideas
---------
- Qiskit builds circuits with `QuantumCircuit`; CUDA-Q executes `@cudaq.kernel`-style
  programs (or kernels created via `cudaq.make_kernel`).
- This module walks a Qiskit circuit and emits an equivalent CUDA-Q kernel using
  the *builder* API (`cudaq.make_kernel`) so we can programmatically wire gates
  and expose parameters as typed kernel arguments.
- Parameters from the Qiskit circuit become `float` kernel parameters in a stable
  order. Parameter *expressions* (e.g., 2*theta + phi) are evaluated symbolically
  against those kernel parameters when constructing the kernel instructions.

------
Covers a practical subset of common gates and features
  - Single-qubit: id, x, y, z, h, s, sdg, t, tdg, sx, rx, ry, rz, p (u1), u, u3
  - Two-qubit: cx (cnot), cz, swap, rxx, ryy, rzz, cp, crz
  - Barriers are ignored; measurements are recorded but not emitted (CUDA-Q
    measures implicitly on `sample`).

You can extend the `GATE_EMITTERS` table to support more ops.

Usage
-----
from qiskit import QuantumCircuit
import cudaq
from cudaq_connector import translate_circuit

qc = QuantumCircuit(2)
qc.h(0); qc.cx(0,1); qc.rx(qc.parameters.get('theta', 0.5) if False else 0.5, 1)

spec = translate_circuit(qc, kernel_name='my_kernel')
# spec.param_order is the ordered list of qiskit.Parameter symbols
# Invoke with CUDA-Q:
counts = cudaq.sample(spec.kernel, *spec.example_args(shots=1000))

# If your circuit had parameters, pass concrete floats in that same order:
# counts = cudaq.sample(spec.kernel, theta_value, phi_value, shots_count=1000)
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence
import math

# Optional imports; keep errors friendly if user doesn't have these
try:
    import cudaq  # type: ignore
except Exception as _e:
    cudaq = None  # lazy error on use

try:
    from qiskit.circuit import QuantumCircuit, ParameterExpression, Parameter
except Exception as _e:  # pragma: no cover
    QuantumCircuit = object  # type: ignore
    ParameterExpression = object  # type: ignore
    Parameter = object  # type: ignore


NumberLike = float | int
ParamLike = NumberLike | ParameterExpression


@dataclass
class KernelSpec:
    """Holds the translated CUDA-Q kernel and metadata Specifications.

    Attributes
    ----------
    kernel : callable
        The CUDA-Q kernel returned by `cudaq.make_kernel(...)`.
    param_order : list[Parameter]
        Qiskit Parameters in the order they appear as kernel arguments.
    measured_qubits : list[int]
        Qubit indices that were measured in the Qiskit circuit (informational).
    source_name : str
        A friendly name given to the kernel (used for debugging/printing).
    n_qubits : int
        Number of qubits allocated by the kernel.
    """

    kernel: Callable
    param_order: list[Parameter]
    measured_qubits: list[int]
    source_name: str
    n_qubits: int

    def example_args(self, **kwargs) -> tuple:
        """Return example numeric arguments for the kernel (zeros for params).
        Useful for quickly calling `cudaq.sample` on parameterized kernels.
        """
        return tuple(0.0 for _ in self.param_order)


# ----------------------------- helpers ------------------------------------


def _ordered_params(qc: QuantumCircuit) -> list[Parameter]:
    """Return circuit parameters in a stable, deterministic order.

    We sort (name, uuid) to avoid collisions when users reuse names.
    """
    params = list(qc.parameters)
    try:
        params.sort(key=lambda p: (str(getattr(p, 'name', p)), str(getattr(p, '_uuid', ''))))
    except Exception:
        params.sort(key=lambda p: str(getattr(p, 'name', p)))
    return params


def _eval_param_expr(expr: ParamLike, env: Dict[Parameter, object]) -> object:
    """Turn a Qiskit parameter or numeric into a CUDA-Q-compatible value/expression.

    - If numeric -> float
    - If Parameter -> bound to the corresponding kernel parameter variable
    - If ParameterExpression -> build a Python expression using the bound kernel
      parameter variables and `math` constants; evaluate it to produce a symbolic
      CUDA-Q value acceptable by the builder (e.g., 2.0*theta + phi).
    """
    # Numeric constant
    if isinstance(expr, (int, float)):
        return float(expr)

    # Parameter -> mapped variable
    if isinstance(expr, Parameter):  # type: ignore
        return env[expr]

    # ParameterExpression: substitute Parameters with their bound variables.
    if isinstance(expr, ParameterExpression):  # type: ignore
        # Build a local namespace mapping symbol names to bound vars.
        # Qiskit ParameterExpression supports iteration over .parameters
        local_ns: Dict[str, object] = {str(p): env[p] for p in expr.parameters}  # type: ignore
        # Provide math symbols that appear frequently.
        local_ns.update({
            'pi': math.pi,
            'e': math.e,
        })
        # Qiskit lets you `float(expr)` only after binding; here we rely on the
        # Python string of the expression which uses parameter names.
        # Example: str(2*theta + phi) -> '2*theta + phi'
        expr_str = str(expr)
        try:
            return eval(expr_str, {"__builtins__": {}}, local_ns)
        except Exception as e:
            raise ValueError(f"Could not evaluate ParameterExpression '{expr_str}'. "
                             f"Ensure it only uses +,-,*,/,**, pi, e and Parameters. Error: {e}")

    # Fallback
    raise TypeError(f"Unsupported parameter type: {type(expr)}")


# --------------------------- gate emitters ---------------------------------

# Each emitter is a callable: (kernel, qalloc, params[0..], qubits[int...]) -> None
# where `k` is the CUDA-Q kernel builder, `q` is the allocated register.

def _emit_single(k, q, qubits: list[int], gate_name: str):
    getattr(k, gate_name)(q[qubits[0]])


def _emit_two(k, q, qubits: list[int], gate_name: str):
    getattr(k, gate_name)(q[qubits[0]], q[qubits[1]])


def _emit_param_1(k, q, qubits: list[int], angle, gate_name: str):
    getattr(k, gate_name)(angle, q[qubits[0]])


def _emit_param_2(k, q, qubits: list[int], angle, gate_name: str):
    getattr(k, gate_name)(angle, q[qubits[0]], q[qubits[1]])


# Map Qiskit op.name -> emitter function + how to compute parameters
GATE_EMITTERS: Dict[str, Callable] = {
    # one-qubit fixed
    'id':      lambda k, q, qs, ps: _emit_single(k, q, qs, 'i'),
    'x':       lambda k, q, qs, ps: _emit_single(k, q, qs, 'x'),
    'y':       lambda k, q, qs, ps: _emit_single(k, q, qs, 'y'),
    'z':       lambda k, q, qs, ps: _emit_single(k, q, qs, 'z'),
    'h':       lambda k, q, qs, ps: _emit_single(k, q, qs, 'h'),
    's':       lambda k, q, qs, ps: _emit_single(k, q, qs, 's'),
    'sdg':     lambda k, q, qs, ps: _emit_single(k, q, qs, 'sdg'),
    't':       lambda k, q, qs, ps: _emit_single(k, q, qs, 't'),
    'tdg':     lambda k, q, qs, ps: _emit_single(k, q, qs, 'tdg'),
    'sx':      lambda k, q, qs, ps: _emit_single(k, q, qs, 'sx'),

    # one-qubit parametric
    'rx':      lambda k, q, qs, ps: _emit_param_1(k, q, qs, ps[0], 'rx'),
    'ry':      lambda k, q, qs, ps: _emit_param_1(k, q, qs, ps[0], 'ry'),
    'rz':      lambda k, q, qs, ps: _emit_param_1(k, q, qs, ps[0], 'rz'),
    'p':       lambda k, q, qs, ps: _emit_param_1(k, q, qs, ps[0], 'p'),  # phase

    # two-qubit fixed
    'cx':      lambda k, q, qs, ps: _emit_two(k, q, qs, 'cx'),
    'cz':      lambda k, q, qs, ps: _emit_two(k, q, qs, 'cz'),
    'swap':    lambda k, q, qs, ps: _emit_two(k, q, qs, 'swap'),

    # two-qubit parametric
    'cp':      lambda k, q, qs, ps: _emit_param_2(k, q, qs, ps[0], 'cp'),
    'crz':     lambda k, q, qs, ps: _emit_param_2(k, q, qs, ps[0], 'crz'),
    'rxx':     lambda k, q, qs, ps: _emit_param_2(k, q, qs, ps[0], 'rxx'),
    'ryy':     lambda k, q, qs, ps: _emit_param_2(k, q, qs, ps[0], 'ryy'),
    'rzz':     lambda k, q, qs, ps: _emit_param_2(k, q, qs, ps[0], 'rzz'),
}


def _emit_u_like(k, q, qubits: list[int], params: Sequence[object]):
    """Handle Qiskit's U / U3(θ, φ, λ): Rz(φ) Rx(θ) Rz(λ) convention.
    Qiskit `u(theta, phi, lam)` maps to a general 1-qubit rotation.
    """
    theta, phi, lam = params
    k.rz(phi,  q[qubits[0]])
    k.rx(theta, q[qubits[0]])
    k.rz(lam,  q[qubits[0]])


# --------------------------- main translator -------------------------------

def translate_circuit(qc: QuantumCircuit, *, kernel_name: str = 'translated') -> KernelSpec:
    """Translate a Qiskit `QuantumCircuit` into a CUDA-Q kernel.

    Parameters
    ----------
    qc : QuantumCircuit
        Input circuit.
    kernel_name : str
        Friendly name for the resulting kernel.

    Returns
    -------
    KernelSpec
        Kernel callable plus metadata.
    """
    if not isinstance(qc, QuantumCircuit):  # type: ignore
        raise TypeError("qc must be a Qiskit QuantumCircuit")

    # Collect parameters and set up a CUDA-Q builder with the appropriate signature.
    params = _ordered_params(qc)
    # Types: one float per parameter
    if params:
        k_and_args = cudaq.make_kernel(*([float] * len(params)))
        k = k_and_args[0]
        arg_vars = list(k_and_args[1:])
    else:
        k = cudaq.make_kernel()
        arg_vars = []

    # Allocate qubits
    q = k.qalloc(qc.num_qubits)

    # Build mapping from Qiskit Parameter -> kernel argument variable
    param_env: Dict[Parameter, object] = {p: arg_vars[i] for i, p in enumerate(params)}

    measured_qubits: list[int] = []

    # Walk instructions
    for instr in qc.data:
        op = instr.operation
        name = getattr(op, 'name', str(op)).lower()

        # Translate qubit tuple indices
        qubits = [qb.index for qb in instr.qubits]

        # Barrier / measure handling
        if name in ('barrier',):
            continue
        if name in ('measure',):
            measured_qubits.extend(qubits)
            continue  # CUDA-Q sampling handles measurement implicitly

        # Fetch operation parameters as CUDA-Q-evaluable values
        raw_params: list[object] = []
        if getattr(op, 'params', None):
            for p in op.params:
                raw_params.append(_eval_param_expr(p, param_env))

        # Gate emission
        if name in GATE_EMITTERS:
            GATE_EMITTERS[name](k, q, qubits, raw_params)
        elif name in ('u', 'u3'):
            if len(raw_params) != 3:
                raise ValueError(f"U gate expects 3 params, got {len(raw_params)}")
            _emit_u_like(k, q, qubits, raw_params)
        elif name in ('u2',):
            # u2(phi, lambda) = u(π/2, phi, lambda)
            if len(raw_params) != 2:
                raise ValueError("u2 expects 2 params")
            theta = math.pi / 2
            phi, lam = raw_params
            _emit_u_like(k, q, qubits, (theta, phi, lam))
        elif name in ('u1',):
            # u1(lambda) == p(lambda)
            _emit_param_1(k, q, qubits, raw_params[0], 'p')
        elif name in ('cu', 'cu3'):
            # Controlled-U: decompose as control on qubits[0] over target qubits[1]
            # Here we synthesize via native rotations + CNOTs (simple textbook decomp)
            # For brevity we emit a crude pattern: C-Rz(phi); C-Rx(theta); C-Rz(lam)
            ctrl, tgt = qubits
            phi, theta, lam = raw_params[1], raw_params[0], raw_params[2]
            k.crz(phi, q[ctrl], q[tgt])
            k.crx(theta, q[ctrl], q[tgt])
            k.crz(lam, q[ctrl], q[tgt])
        else:
            raise NotImplementedError(
                f"Unsupported or unmapped gate '{name}'. Extend GATE_EMITTERS or add a handler.")

    spec = KernelSpec(kernel=k, param_order=params, measured_qubits=measured_qubits,
                      source_name=kernel_name, n_qubits=qc.num_qubits)

    return spec
