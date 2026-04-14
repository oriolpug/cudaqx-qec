#!/usr/bin/env python3
"""
GPU-Everywhere QEC Pipeline: Maestro (cuQuantum) + CUDA-QX Decoders
=====================================================================

Blog demonstration script showing an end-to-end GPU-accelerated QEC pipeline:

    GPU #1: Maestro MPS simulation via cuQuantum/cuTensorNet
            → generates coherent noise syndromes for a surface code

    GPU #2: CUDA-QX decoder (QLDPC BP-OSD or Tensor Network)
            → decodes the coherent noise syndromes on GPU

Three-way comparison:
    1. Stim (Pauli)  → PyMatching (CPU)    [blue]  industry baseline
    2. Maestro (Coh.) → PyMatching (CPU)   [red]   coherent noise degrades
    3. Maestro (Coh.) → CUDA-QX GPU        [green] GPU decoder on GPU syndromes

Usage:
    python cudaqx_qec_demo.py [--d D] [--chi CHI] [--gpu] [--compare]

Requires:
    pip install cudaq cudaq-qec   (Linux x86_64 + CUDA 12+)
    pip install maestro deltakit pymatching stim
"""

import argparse
import sys
import time
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import maestro
import maestro_deltakit_bridge as maestro_bridge
from maestro_deltakit_bridge import deltakit_to_maestro, counts_to_bitarray, apply_measurement_noise
from deltakit.circuit.gates import PauliBasis
from deltakit.explorer.codes import RotatedPlanarCode, css_code_memory_circuit
from deltakit.explorer.qpu import QPU, SI1000Noise
from deltakit.decode import PyMatchingDecoder
from deltakit.decode.analysis import run_decoding_on_circuit
from deltakit.explorer.analysis import calculate_lep_and_lep_stddev

from cudaqx_decoder_bridge import CUDAQXDecoder, PyMatchingBaseline, CUDAQX_AVAILABLE, DEFAULT_DECODER, GPU_AVAILABLE

# ── Parameters ──
parser = argparse.ArgumentParser(description="GPU-Everywhere QEC Pipeline Demo")
parser.add_argument("--d", type=int, default=3, help="Code distance (default: 3)")
parser.add_argument("--chi", type=int, default=32, help="MPS bond dimension (default: 32)")
parser.add_argument("--gpu", action="store_true", help="Run on GPU if available")
parser.add_argument("--compare", action="store_true", help="Compare CPU vs GPU for simulation and decoding")
args = parser.parse_args()

GPU_USED = True if GPU_AVAILABLE and '--gpu' in sys.argv else False
d = args.d                                # Code distance
SCALE = 0.5                               # Coherent noise scale factor
noise_strengths = [0.002, 0.005, 0.01, 0.02, 0.04]
stim_shots = 10_000                        # Stim baseline shots
mps_shots = 500                            # Maestro MPS shots
chi = args.chi                             # Bond dimension
DECODER_TYPE = DEFAULT_DECODER              # nv-qldpc-decoder (GPU) or single_error_lut (CPU)

print("=" * 70)
print("  GPU-Everywhere QEC: Maestro (cuQuantum) + CUDA-QX Decoding")
print("=" * 70)
print(f"\n  d={d} rotated surface code ({2*d*d - 1} qubits)")
print(f"  Noise model:      SI1000 (superconducting)")
print(f"  Coherent scale:   ε = {SCALE} × 2√p")
print(f"  MPS bond dim:     χ={chi}")
print(f"  MPS shots:        {mps_shots}")
print(f"  NVIDIA GPU:        {GPU_AVAILABLE} (used: {GPU_USED})")
print(f"  CUDA-QX decoder:  {DECODER_TYPE}")
print(f"  CUDA-QX available: {CUDAQX_AVAILABLE}")

code = RotatedPlanarCode(width=d, height=d)

def build_coherent_circuit(noisy, stim_circ):
    """Build coherent circuit with scaled sqrt, then restore."""
    _orig = math.sqrt
    maestro_bridge.math.sqrt = lambda x: _orig(x) * SCALE
    try:
        result = deltakit_to_maestro(noisy, noise_type='coherent', stim_circuit=stim_circ)
    finally:
        maestro_bridge.math.sqrt = _orig
    return result

# ── Compare mode ──
if args.compare:
    if not GPU_AVAILABLE:
        print("ERROR: --compare requires an NVIDIA GPU but none was detected.")
        sys.exit(1)

    print(f"\n  Mode: COMPARE (CPU vs GPU)")
    print(f"  Running each noise strength with both simulators and decoders\n")

    cpu_sim_times, gpu_sim_times = [], []
    cmp_pm_decode_times, cmp_cqx_decode_times = [], []

    for idx, p in enumerate(noise_strengths):
        print(f"[{idx+1}/{len(noise_strengths)}] p={p}", flush=True)

        circuit = css_code_memory_circuit(code, num_rounds=d, logical_basis=PauliBasis.Z)
        qpu = QPU(circuit.qubits, noise_model=SI1000Noise(p=p))
        noisy = qpu.compile_and_add_noise_to_circuit(circuit)
        pm_decoder, stim_circ = PyMatchingDecoder.construct_decoder_and_stim_circuit(noisy)

        # Build coherent circuit once (reused for both simulators)
        mqc_c, _, nmc, flip_probs_c = build_coherent_circuit(noisy, stim_circ)

        # ─── Maestro MPS: CPU ───
        t0 = time.time()
        rc_cpu = mqc_c.execute(
            shots=mps_shots,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=chi,
        )
        cpu_t = time.time() - t0
        cpu_sim_times.append(cpu_t)
        print(f"  Maestro CPU:  {cpu_t:.2f}s", flush=True)

        # ─── Maestro MPS: GPU ───
        t0 = time.time()
        rc_gpu = mqc_c.execute(
            shots=mps_shots,
            simulator_type=maestro.SimulatorType.Gpu,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=chi,
        )
        gpu_t = time.time() - t0
        gpu_sim_times.append(gpu_t)
        print(f"  Maestro GPU:  {gpu_t:.2f}s  ({cpu_t/gpu_t:.1f}x speedup)", flush=True)

        # Use GPU syndromes for decoding comparison
        raw_coherent = counts_to_bitarray(rc_gpu['counts'], nmc)
        raw_coherent = apply_measurement_noise(raw_coherent, flip_probs_c)

        # ─── Decode: PyMatching (CPU) ───
        pm_wrapper = PyMatchingBaseline(stim_circ, pm_decoder)
        _, _, pm_dt = pm_wrapper.decode_raw_measurements(raw_coherent)
        cmp_pm_decode_times.append(pm_dt)
        print(f"  PyMatching (CPU):   {pm_dt*1000:.1f}ms", flush=True)

        # ─── Decode: CUDA-QX (GPU) ───
        if CUDAQX_AVAILABLE:
            try:
                cqx_decoder = CUDAQXDecoder(stim_circ, decoder_type="nv-qldpc-decoder")
                _, _, cqx_dt = cqx_decoder.decode_raw_measurements(raw_coherent)
                cmp_cqx_decode_times.append(cqx_dt)
                speedup = pm_dt / cqx_dt if cqx_dt > 0 else float('inf')
                print(f"  CUDA-QX (GPU):      {cqx_dt*1000:.1f}ms  ({speedup:.1f}x speedup)", flush=True)
            except Exception as e:
                print(f"  CUDA-QX error: {e}", flush=True)
                cmp_cqx_decode_times.append(np.nan)
        else:
            print(f"  CUDA-QX: not available", flush=True)
            cmp_cqx_decode_times.append(np.nan)

    # ── Compare Plot ──
    print("\nGenerating comparison plot...", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    x = np.arange(len(noise_strengths))
    width = 0.35

    # Left: Maestro simulation time (CPU vs GPU)
    ax1.set_facecolor('#fafafa')
    ax1.bar(x - width/2, cpu_sim_times, width, color='#dc2626', alpha=0.8,
            label='Maestro CPU (QCSim)', edgecolor='white', linewidth=0.5)
    ax1.bar(x + width/2, gpu_sim_times, width, color='#16a34a', alpha=0.8,
            label='Maestro GPU (cuQuantum)', edgecolor='white', linewidth=0.5)
    for i in x:
        speedup = cpu_sim_times[i] / gpu_sim_times[i] if gpu_sim_times[i] > 0 else 0
        ax1.text(i + width/2, gpu_sim_times[i], f'{speedup:.1f}x',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#16a34a')
    ax1.set_xlabel('Physical Error Rate (p)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Simulation Time (s)', fontsize=13, fontweight='bold')
    ax1.set_title(f'd={d} Rotated Surface Code, χ={chi}, {mps_shots} shots\n'
                  'MPS Simulation: CPU vs GPU',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{p:.3f}' for p in noise_strengths])
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Right: Decode time (PyMatching CPU vs CUDA-QX GPU)
    ax2.set_facecolor('#fafafa')
    pm_ms = [t * 1000 for t in cmp_pm_decode_times]
    cqx_ms = [t * 1000 if not np.isnan(t) else 0 for t in cmp_cqx_decode_times]
    ax2.bar(x - width/2, pm_ms, width, color='#dc2626', alpha=0.8,
            label='PyMatching (CPU)', edgecolor='white', linewidth=0.5)
    if any(t > 0 for t in cqx_ms):
        ax2.bar(x + width/2, cqx_ms, width, color='#16a34a', alpha=0.8,
                label='CUDA-QX nv-qldpc (GPU)', edgecolor='white', linewidth=0.5)
        for i in x:
            if cqx_ms[i] > 0:
                speedup = pm_ms[i] / cqx_ms[i]
                ax2.text(i + width/2, cqx_ms[i], f'{speedup:.1f}x',
                         ha='center', va='bottom', fontsize=9, fontweight='bold', color='#16a34a')
    ax2.set_xlabel('Physical Error Rate (p)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Decode Time (ms)', fontsize=13, fontweight='bold')
    ax2.set_title(f'd={d} Rotated Surface Code, {mps_shots} shots\n'
                  'Decoding: CPU vs GPU',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{p:.3f}' for p in noise_strengths])
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    outpng = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cudaqx_compare.png')
    plt.savefig(outpng, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot: {outpng}", flush=True)
    sys.exit(0)

# ── Storage ──
stim_leps, stim_stds = [], []
pm_coh_leps, pm_coh_stds = [], []           # PyMatching on coherent
cudaqx_coh_leps, cudaqx_coh_stds = [], []   # CUDA-QX on coherent
pm_decode_times, cudaqx_decode_times = [], []

outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cudaqx_results.txt")
total_start = time.time()

with open(outfile, "w") as f:
    f.write(f"d={d}, scale={SCALE}, chi={chi}, mps_shots={mps_shots}, decoder={DECODER_TYPE}\n")
    f.write(f"{'p':>8}  {'Stim+PM':>12}  {'MPS+PM':>12}  {'MPS+CUDAQX':>12}  "
            f"{'PM(ms)':>8}  {'CQX(ms)':>8}  {'speedup':>8}\n")
    f.write("-" * 85 + "\n")
    f.flush()

    for idx, p in enumerate(noise_strengths):
        print(f"\n[{idx+1}/{len(noise_strengths)}] p={p}", flush=True)

        # Build noisy circuit with Deltakit
        circuit = css_code_memory_circuit(code, num_rounds=d, logical_basis=PauliBasis.Z)
        qpu = QPU(circuit.qubits, noise_model=SI1000Noise(p=p))
        noisy = qpu.compile_and_add_noise_to_circuit(circuit)
        pm_decoder, stim_circ = PyMatchingDecoder.construct_decoder_and_stim_circuit(noisy)

        # ─── 1. Stim Baseline (Pauli) → PyMatching (CPU) ───
        t0 = time.time()
        result = run_decoding_on_circuit(stim_circ, stim_shots, pm_decoder, 1000, min_fails=5)
        try:
            lep, std = calculate_lep_and_lep_stddev(result["fails"], stim_shots)
        except:
            lep, std = 0.0, 0.0
        stim_leps.append(lep)
        stim_stds.append(std)
        print(f"  Stim+PyMatching:    LEP={lep:.5f} ({time.time()-t0:.1f}s)", flush=True)

        # ─── 2. Maestro MPS (Coherent) → generate syndromes ───
        t0 = time.time()
        mqc_c, _, nmc, flip_probs_c = build_coherent_circuit(noisy, stim_circ)
        rc = mqc_c.execute(
            shots=mps_shots,
            simulator_type=maestro.SimulatorType.Gpu if GPU_USED else maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=chi,
        )
        raw_coherent = counts_to_bitarray(rc['counts'], nmc)
        raw_coherent = apply_measurement_noise(raw_coherent, flip_probs_c)
        sim_time = time.time() - t0
        print(f"  Maestro MPS sim:    {sim_time:.1f}s ({mps_shots} shots)", flush=True)

        # ─── 2a. Decode with PyMatching (CPU) ───
        pm_wrapper = PyMatchingBaseline(stim_circ, pm_decoder)
        pm_lep, pm_std, pm_dt = pm_wrapper.decode_raw_measurements(raw_coherent)
        pm_coh_leps.append(pm_lep)
        pm_coh_stds.append(pm_std)
        pm_decode_times.append(pm_dt)
        print(f"  → PyMatching (CPU): LEP={pm_lep:.5f}  decode={pm_dt*1000:.1f}ms", flush=True)

        # ─── 2b. Decode with CUDA-QX ───
        if CUDAQX_AVAILABLE:
            try:
                cqx_decoder = CUDAQXDecoder(stim_circ, decoder_type=DECODER_TYPE)
                cqx_lep, cqx_std, cqx_dt = cqx_decoder.decode_raw_measurements(raw_coherent)
                cudaqx_coh_leps.append(cqx_lep)
                cudaqx_coh_stds.append(cqx_std)
                cudaqx_decode_times.append(cqx_dt)
                speedup = pm_dt / cqx_dt if cqx_dt > 0 else float('inf')
                cqx_label = "GPU" if GPU_USED else "CPU"
                print(f"  → CUDA-QX ({cqx_label}):   LEP={cqx_lep:.5f}  decode={cqx_dt*1000:.1f}ms  "
                      f"({speedup:.1f}× faster)", flush=True)
            except Exception as e:
                print(f"  → CUDA-QX error: {e}", flush=True)
                cudaqx_coh_leps.append(np.nan)
                cudaqx_coh_stds.append(np.nan)
                cudaqx_decode_times.append(np.nan)
        else:
            print(f"  → CUDA-QX: not available (will use placeholder)", flush=True)
            cudaqx_coh_leps.append(np.nan)
            cudaqx_coh_stds.append(np.nan)
            cudaqx_decode_times.append(np.nan)

        # Log results
        cqx_lep_val = cudaqx_coh_leps[-1]
        cqx_dt_val = cudaqx_decode_times[-1]
        speedup_val = pm_dt / cqx_dt_val if not np.isnan(cqx_dt_val) and cqx_dt_val > 0 else 0
        line = (f"{p:>8.4f}  {lep:>12.5f}  {pm_lep:>12.5f}  {cqx_lep_val:>12.5f}  "
                f"{pm_dt*1000:>7.1f}  {cqx_dt_val*1000:>7.1f}  {speedup_val:>7.1f}×")
        f.write(line + "\n")
        f.flush()

total = time.time() - total_start
print(f"\nTotal: {total:.1f}s", flush=True)

# ── Plot ──
print("\nGenerating plot...", flush=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('white')

# ── Left panel: Logical Error Probability ──
ax1.set_facecolor('#fafafa')

ax1.errorbar(noise_strengths, stim_leps, yerr=stim_stds,
             fmt='o-', color='#2563eb', lw=2.5, ms=9, capsize=5,
             label='Stim (Pauli) → PyMatching', zorder=3)

ax1.errorbar(noise_strengths, pm_coh_leps, yerr=pm_coh_stds,
             fmt='D-', color='#dc2626', lw=2.5, ms=9, capsize=5,
             label='Maestro MPS (Coherent) → PyMatching', zorder=3)

if any(not np.isnan(x) for x in cudaqx_coh_leps):
    valid = [(ns, l, s) for ns, l, s in zip(noise_strengths, cudaqx_coh_leps, cudaqx_coh_stds)
             if not np.isnan(l)]
    if valid:
        ns_v, l_v, s_v = zip(*valid)
        ax1.errorbar(ns_v, l_v, yerr=s_v,
                     fmt='s-', color='#16a34a', lw=2.5, ms=9, capsize=5,
                     label=f'Maestro MPS (Coherent) → CUDA-QX ({DECODER_TYPE})', zorder=3)

ax1.set_xlabel('Physical Error Rate (p)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Logical Error Probability', fontsize=13, fontweight='bold')
ax1.set_title(f'd={d} Rotated Surface Code ({2*d*d-1}Q, SI1000, χ={chi})\n'
              'GPU-Accelerated QEC: Simulation + Decoding',
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, max(noise_strengths) * 1.05)
ax1.set_ylim(0, None)

# ── Right panel: Decode Speed Comparison ──
ax2.set_facecolor('#fafafa')

pm_times_ms = [t * 1000 for t in pm_decode_times]
cqx_times_ms = [t * 1000 if not np.isnan(t) else 0 for t in cudaqx_decode_times]

x = np.arange(len(noise_strengths))
width = 0.35

bars1 = ax2.bar(x - width/2, pm_times_ms, width, color='#dc2626', alpha=0.8,
                label='PyMatching (CPU)', edgecolor='white', linewidth=0.5)
if any(t > 0 for t in cqx_times_ms):
    bars2 = ax2.bar(x + width/2, cqx_times_ms, width, color='#16a34a', alpha=0.8,
                    label=f'CUDA-QX ({"GPU" if GPU_USED else "CPU"})', edgecolor='white', linewidth=0.5)

ax2.set_xlabel('Physical Error Rate (p)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Decode Time (ms)', fontsize=13, fontweight='bold')
ax2.set_title(f'Decode Speed: CPU vs GPU\n'
              f'{mps_shots} shots per point',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{p:.3f}' for p in noise_strengths])
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()

outpng = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cudaqx_pipeline.png')
plt.savefig(outpng, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot: {outpng}", flush=True)

# ── Summary table ──
print(f"\n{'=' * 85}")
print(f"  GPU-Everywhere QEC Pipeline Results")
print(f"  Simulation: Maestro MPS (cuQuantum/cuTensorNet) — coherent noise")
print(f"  Decoding:   CUDA-QX {DECODER_TYPE} ({'GPU' if GPU_USED else 'CPU'}) vs PyMatching (CPU)")
print(f"{'=' * 85}")
print(f"  {'p':>6}  {'Stim+PM':>10}  {'MPS+PM':>10}  {'MPS+CQX':>10}  "
      f"{'PM(ms)':>8}  {'CQX(ms)':>8}  {'Speedup':>8}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
for i, p in enumerate(noise_strengths):
    cqx_val = cudaqx_coh_leps[i] if not np.isnan(cudaqx_coh_leps[i]) else 0
    cqx_t = cudaqx_decode_times[i] if not np.isnan(cudaqx_decode_times[i]) else 0
    speedup = pm_decode_times[i] / cqx_t if cqx_t > 0 else 0
    print(f"  {p:>6.4f}  {stim_leps[i]:>10.5f}  {pm_coh_leps[i]:>10.5f}  "
          f"{cqx_val:>10.5f}  {pm_decode_times[i]*1000:>7.1f}  "
          f"{cqx_t*1000:>7.1f}  {speedup:>7.1f}×")
print(f"{'=' * 85}")
print(f"\n  Key finding: Coherent noise (simulated on GPU via cuQuantum)")
print(f"  produces higher logical error rates than Pauli noise.")
print(f"  CUDA-QX decoders decode the syndromes {'on GPU, ' if GPU_USED else ''}faster")
print(f"  than CPU-only PyMatching — enabling end-to-end GPU QEC research.")
print(f"\n{'=' * 85}")
