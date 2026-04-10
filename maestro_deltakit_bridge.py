"""
Maestro-Deltakit Bridge — Stim/Deltakit QEC Circuits → Maestro QuantumCircuit
==============================================================================

Converts deltakit noisy QEC circuits (via their stim representation) into
maestro QuantumCircuit objects for GPU-accelerated coherent noise simulation.

In the coherent noise model, stochastic Pauli error channels (DEPOLARIZE1,
DEPOLARIZE2, X_ERROR) are replaced by deterministic unitary rotations with
angle epsilon = 2*sqrt(p), matching the average gate infidelity of the
original channel.

Three public functions:
    deltakit_to_maestro  — circuit conversion with coherent noise injection
    counts_to_bitarray   — maestro counts dict → numpy measurement array
    apply_measurement_noise — post-hoc classical bit-flip noise
"""

import math
import warnings

import numpy as np
import maestro

_SKIP = frozenset({
    'TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE',
    'QUBIT_COORDS', 'SHIFT_COORDS',
})


def deltakit_to_maestro(noisy_circuit, noise_type='coherent', stim_circuit=None):
    """
    Convert a deltakit/stim noisy QEC circuit to a maestro QuantumCircuit.

    Iterates the flattened stim circuit, maps gate operations to maestro
    equivalents, and replaces Pauli noise channels with coherent unitary
    rotations (Rx then Rz with angle 2*sqrt(p)) when noise_type='coherent'.

    Parameters
    ----------
    noisy_circuit : deltakit Circuit
        The noisy deltakit circuit (used as fallback if stim_circuit is None).
    noise_type : str
        'coherent' to replace noise channels with unitary rotations,
        anything else to produce a noiseless circuit.
    stim_circuit : stim.Circuit, optional
        Pre-built stim circuit. If None, derived from noisy_circuit.

    Returns
    -------
    (qc, num_qubits, num_measurements, flip_probs) : tuple
        qc              — maestro QuantumCircuit ready for execute()
        num_qubits      — number of qubits in the circuit
        num_measurements — total number of measurement outcomes
        flip_probs      — (num_measurements,) array of measurement flip probs
    """
    if stim_circuit is None:
        stim_circuit = noisy_circuit.as_stim_circuit()

    coherent = (noise_type == 'coherent')

    qc = maestro.circuits.QuantumCircuit()
    cbit = 0
    flip_probs = []

    for inst in stim_circuit.flattened():
        name = inst.name

        if name in _SKIP:
            continue

        targets = inst.targets_copy()
        args = inst.gate_args_copy()

        # ── Gate operations ──
        if name == 'R':
            for t in targets:
                qc.reset(t.value)

        elif name == 'H':
            for t in targets:
                qc.h(t.value)

        elif name == 'X':
            for t in targets:
                qc.x(t.value)

        elif name == 'Y':
            for t in targets:
                qc.y(t.value)

        elif name == 'Z':
            for t in targets:
                qc.z(t.value)

        elif name == 'S':
            for t in targets:
                qc.s(t.value)

        elif name == 'S_DAG':
            for t in targets:
                qc.sdg(t.value)

        elif name == 'CX':
            for i in range(0, len(targets), 2):
                qc.cx(targets[i].value, targets[i + 1].value)

        elif name == 'CZ':
            for i in range(0, len(targets), 2):
                qc.cz(targets[i].value, targets[i + 1].value)

        # ── Measurements ──
        elif name == 'M':
            p = args[0] if args else 0.0
            meas_pairs = []
            for t in targets:
                meas_pairs.append((t.value, cbit))
                flip_probs.append(p)
                cbit += 1
            qc.measure(meas_pairs)

        elif name == 'MR':
            p = args[0] if args else 0.0
            for t in targets:
                qc.measure([(t.value, cbit)])
                flip_probs.append(p)
                cbit += 1
                qc.reset(t.value)

        # ── Noise channels → coherent rotations ──
        elif name == 'X_ERROR':
            if coherent:
                angle = 2.0 * math.sqrt(args[0])
                for t in targets:
                    qc.rx(t.value, angle)

        elif name == 'DEPOLARIZE1':
            if coherent:
                angle = 2.0 * math.sqrt(args[0])
                for t in targets:
                    qc.rx(t.value, angle)
                    qc.rz(t.value, angle)

        elif name == 'DEPOLARIZE2':
            if coherent:
                angle = 2.0 * math.sqrt(args[0])
                for i in range(0, len(targets), 2):
                    for q in (targets[i].value, targets[i + 1].value):
                        qc.rx(q, angle)
                        qc.rz(q, angle)

        elif name in ('Y_ERROR', 'Z_ERROR'):
            if coherent:
                angle = 2.0 * math.sqrt(args[0])
                for t in targets:
                    if name == 'Y_ERROR':
                        qc.ry(t.value, angle)
                    else:
                        qc.rz(t.value, angle)

        else:
            warnings.warn(
                f"maestro_deltakit_bridge: skipping unrecognized "
                f"stim instruction '{name}'"
            )

    return qc, stim_circuit.num_qubits, cbit, np.array(flip_probs)


def counts_to_bitarray(counts_dict, n_meas_cols):
    """
    Convert maestro counts dictionary to a measurement bit-array.

    Parameters
    ----------
    counts_dict : dict[str, int]
        Mapping from bitstring (e.g. '0110') to shot count.
    n_meas_cols : int
        Number of measurement columns (expected bitstring length).

    Returns
    -------
    np.ndarray of shape (n_shots, n_meas_cols), dtype uint8
    """
    if not counts_dict:
        return np.zeros((0, n_meas_cols), dtype=np.uint8)

    rows = []
    counts = []
    for bitstring, count in counts_dict.items():
        row = np.array(
            [int(c) for c in bitstring[:n_meas_cols]],
            dtype=np.uint8,
        )
        if len(row) < n_meas_cols:
            row = np.pad(row, (0, n_meas_cols - len(row)))
        rows.append(row)
        counts.append(count)

    unique_rows = np.array(rows, dtype=np.uint8)
    return np.repeat(unique_rows, counts, axis=0)


def apply_measurement_noise(bitarray, flip_probs):
    """
    Apply post-hoc classical bit-flip noise to a measurement array.

    For each measurement column j, each bit is independently flipped
    with probability flip_probs[j].

    Parameters
    ----------
    bitarray : np.ndarray of shape (n_shots, n_meas)
        Clean measurement outcomes (uint8, 0 or 1).
    flip_probs : np.ndarray of shape (n_meas,)
        Per-measurement flip probability.

    Returns
    -------
    np.ndarray of same shape, dtype uint8
    """
    n_cols = bitarray.shape[1]
    probs = flip_probs[:n_cols]
    flip_mask = (np.random.random(bitarray.shape) < probs[np.newaxis, :])
    return bitarray ^ flip_mask.astype(np.uint8)
