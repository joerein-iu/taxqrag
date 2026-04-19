"""
quantum/sampler_ibm.py

IBM Quantum sampler for QRAG — replaces D-Wave sampler.

D-Wave uses quantum annealing — physical energy minimization.
IBM uses gate-based QAOA — a variational quantum algorithm
that approximates combinatorial optimization on gate hardware.

The QUBO objective is identical. Only the solver backend changes.
This is the key difference documented in the paper:
  Section 4.2 — Hardware Backends
  "We evaluate both quantum annealing (D-Wave) and gate-based
   QAOA (IBM Quantum) as solvers for the document selection QUBO."
"""

import numpy as np
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
from quantum.qubo_tax import build_tax_qubo, decode_tax_solution, greedy_tax_fallback
from quantum.interactions import score_combination
from config import (
    TOP_K_FINAL,
    MAX_TOKENS_CONTEXT,
    IBM_QUANTUM_API_KEY as IBM_API_KEY,
    IBM_QUANTUM_CRN as IBM_CRN,
    IBM_QUANTUM_CHANNEL as IBM_CHANNEL,
)


def qubo_to_ising(bqm: BinaryQuadraticModel) -> tuple:
    """
    Convert QUBO (Binary variables) to Ising model (±1 spin variables).
    Required for QAOA circuits which operate on spin/qubit variables.

    Transformation: x_i = (1 - z_i) / 2
    where x_i ∈ {0,1} and z_i ∈ {-1, +1}

    Returns: (h, J, offset)
      h: dict of linear Ising coefficients
      J: dict of quadratic Ising coefficients
      offset: constant energy offset
    """
    ising = bqm.spin  # dimod handles the conversion
    h = dict(ising.linear)
    J = dict(ising.quadratic)
    offset = ising.offset
    return h, J, offset


def build_qaoa_circuit(h: dict, J: dict, num_qubits: int, reps: int = 2):
    """
    Build a QAOA circuit for the Ising problem.

    QAOA alternates between:
    1. Problem (phase) layer: encodes the objective function
    2. Mixer layer: explores the solution space

    reps: number of QAOA layers (higher = better approximation, more circuit depth)
    For NISQ hardware, reps=1 or 2 is practical.
    """
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp

    # Build cost Hamiltonian from Ising coefficients
    pauli_list = []

    # Linear terms: h_i * Z_i
    for qubit, coeff in h.items():
        if abs(coeff) > 1e-10:
            pauli_str = ['I'] * num_qubits
            pauli_str[qubit] = 'Z'
            pauli_list.append((''.join(reversed(pauli_str)), coeff))

    # Quadratic terms: J_ij * Z_i * Z_j
    for (qi, qj), coeff in J.items():
        if abs(coeff) > 1e-10:
            pauli_str = ['I'] * num_qubits
            pauli_str[qi] = 'Z'
            pauli_str[qj] = 'Z'
            pauli_list.append((''.join(reversed(pauli_str)), coeff))

    if not pauli_list:
        # Trivial problem — return simple circuit
        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))
        return qc, SparsePauliOp.from_list([('I' * num_qubits, 1.0)])

    cost_hamiltonian = SparsePauliOp.from_list(pauli_list)
    qaoa = QAOAAnsatz(cost_hamiltonian, reps=reps)

    return qaoa, cost_hamiltonian


def solve_ibm_quantum(
    candidates: list[dict],
    k: int = TOP_K_FINAL,
    token_budget: int = MAX_TOKENS_CONTEXT,
    use_real_hardware: bool = True,
    shots: int = 1024,
    qaoa_seed: int = 42,
    optimization_level: int = 3,
) -> dict:
    """
    Solve document selection QUBO using IBM Quantum hardware via QAOA.

    For the paper: this function is called with use_real_hardware=True
    for the "IBM Quantum" column in Table 1, and use_real_hardware=False
    for the "QAOA Simulator" column.
    """
    start = time.time()
    method = None

    print(f"Building QUBO for {len(candidates)} candidates...")
    bqm = build_tax_qubo(candidates, k=k, token_budget=token_budget)

    # Limit qubit count for NISQ hardware
    # Full problem may exceed free tier qubit limits
    # We take the top-N candidates by score to keep circuit tractable
    MAX_QUBITS = 20  # safe for IBM Open Plan hardware
    if len(candidates) > MAX_QUBITS:
        print(f"Truncating to {MAX_QUBITS} candidates for hardware compatibility")
        candidates_hw = sorted(candidates, key=lambda x: x['score'], reverse=True)[:MAX_QUBITS]
        bqm = build_tax_qubo(candidates_hw, k=k, token_budget=token_budget)
    else:
        candidates_hw = candidates

    num_qubits = len(candidates_hw)
    h, J, offset = qubo_to_ising(bqm)

    sample = None
    energy = None
    hardware_extras: dict = {}

    if use_real_hardware and IBM_API_KEY:
        try:
            sample, energy, method, hardware_extras = _solve_ibm_hardware(
                h, J, bqm, num_qubits, shots=shots,
                qaoa_seed=qaoa_seed,
                optimization_level=optimization_level,
            )
        except Exception as e:
            print(f"IBM Quantum hardware failed: {e}")
            print("Falling back to QAOA simulator...")

    if sample is None:
        try:
            sample, energy, method = _solve_qaoa_simulator(
                h, J, bqm, num_qubits, shots=shots, qaoa_seed=qaoa_seed,
            )
        except Exception as e:
            print(f"QAOA simulator failed: {e}, using simulated annealing")
            sample, energy, method = _solve_simulated_annealing(bqm)

    solve_time_ms = (time.time() - start) * 1000

    # Decode
    if sample:
        selected = decode_tax_solution(sample, candidates_hw)
    else:
        selected = greedy_tax_fallback(candidates_hw, k=k, token_budget=token_budget)
        method = "greedy_fallback"

    if not selected:
        selected = candidates_hw[:k]

    print(f"Solver: {method} | {solve_time_ms:.0f}ms | Selected: {len(selected)} docs")

    return {
        "selected_docs": selected,
        "method": method,
        "solve_time_ms": solve_time_ms,
        "energy": energy,
        "num_candidates": len(candidates),
        "num_qubits_used": num_qubits,
        "hardware_extras": hardware_extras,
    }


def _solve_ibm_hardware(h, J, bqm, num_qubits, shots=1024, qaoa_seed=42, optimization_level=3):
    """Submit QAOA to real IBM Quantum hardware."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    print(f"Connecting to IBM Quantum via channel={IBM_CHANNEL}...")
    service = QiskitRuntimeService(
        channel=IBM_CHANNEL,
        token=IBM_API_KEY,
        instance=IBM_CRN,
    )

    # Get least busy backend with enough qubits
    backend = service.least_busy(
        min_num_qubits=num_qubits,
        operational=True
    )
    print(f"Selected backend: {backend.name} ({backend.num_qubits} qubits)")

    # Build and optimize QAOA circuit
    qaoa_circuit, cost_hamiltonian = build_qaoa_circuit(h, J, num_qubits, reps=1)

    # Use fixed parameters for simplicity
    # (production: use VQE optimizer to find optimal params)
    import numpy as np
    num_params = len(qaoa_circuit.parameters)
    rng = np.random.default_rng(qaoa_seed)
    optimal_params = rng.uniform(0, np.pi, num_params)
    bound_circuit = qaoa_circuit.assign_parameters(optimal_params)
    bound_circuit.measure_all()

    # Transpile for target hardware
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_circuit = pm.run(bound_circuit)

    print(f"Circuit depth: {isa_circuit.depth()} | Submitting {shots} shots...")

    sampler = Sampler(backend)
    job = sampler.run([isa_circuit], shots=shots)
    job_id = job.job_id()
    print(f"Job ID: {job_id} — polling status (5-min queue cap)...")

    # Poll with a 5-minute queue cap. If still queued/initializing after
    # that, cancel and raise so the caller falls back to Aer.
    queue_timeout_s = 300
    poll_start = time.time()
    terminal = {"DONE", "CANCELLED", "ERROR"}
    while True:
        status = str(job.status())
        elapsed = time.time() - poll_start
        print(f"  [{elapsed:5.0f}s] status={status}")
        if status in terminal or status == "JobStatus.DONE":
            break
        if elapsed > queue_timeout_s:
            print(f"Queue time exceeded {queue_timeout_s}s — cancelling job {job_id}")
            try:
                job.cancel()
            except Exception as cancel_err:
                print(f"  cancel raised: {cancel_err}")
            raise TimeoutError(
                f"IBM job {job_id} exceeded {queue_timeout_s}s queue cap"
            )
        time.sleep(10)

    result = job.result()

    # Extract best bitstring
    counts = result[0].data.meas.get_counts()
    best_bitstring = max(counts, key=counts.get)

    # Convert bitstring to sample dict
    sample = {}
    for i, bit in enumerate(reversed(best_bitstring)):
        sample[i] = int(bit)

    # Compute energy of best solution
    energy = bqm.energy(sample)

    extras = {
        "circuit_depth": isa_circuit.depth(),
        "job_id": job_id,
        "backend": backend.name,
        "optimization_level": optimization_level,
        "qaoa_seed": qaoa_seed,
    }
    return sample, energy, f"ibm_quantum_{backend.name}", extras


def _solve_qaoa_simulator(h, J, bqm, num_qubits, shots=1024, qaoa_seed=42):
    """Run QAOA on local Aer simulator — no quantum time consumed."""
    from qiskit_aer import AerSimulator
    import numpy as np

    print(f"Running QAOA on local Aer simulator (seed={qaoa_seed})...")
    simulator = AerSimulator()

    qaoa_circuit, cost_hamiltonian = build_qaoa_circuit(h, J, num_qubits, reps=2)

    num_params = len(qaoa_circuit.parameters)
    rng = np.random.default_rng(qaoa_seed)
    params = rng.uniform(0, np.pi, num_params)
    bound_circuit = qaoa_circuit.assign_parameters(params)
    bound_circuit.measure_all()

    from qiskit import transpile
    compiled = transpile(bound_circuit, simulator, seed_transpiler=qaoa_seed)
    job = simulator.run(compiled, shots=shots, seed_simulator=qaoa_seed)
    result = job.result()
    counts = result.get_counts()

    best_bitstring = max(counts, key=counts.get)
    sample = {i: int(bit) for i, bit in enumerate(reversed(best_bitstring))}
    energy = bqm.energy(sample)

    return sample, energy, "qaoa_simulator"


def _solve_simulated_annealing(bqm):
    """Classical simulated annealing fallback."""
    sa = SimulatedAnnealingSampler()
    response = sa.sample(bqm, num_reads=50, num_sweeps=500)
    best = response.first
    return dict(best.sample), best.energy, "simulated_annealing"
