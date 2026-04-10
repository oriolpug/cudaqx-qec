"""
CUDA-QX Decoder Bridge — Maestro Syndromes → GPU-Accelerated Decoding
======================================================================

Connects Maestro's coherent noise simulation output to NVIDIA CUDA-QX's
GPU-accelerated QEC decoders (QLDPC BP-OSD, Tensor Network, or TensorRT).

This bridge is the "GPU #2" in our pipeline:
    GPU #1: Maestro MPS (cuQuantum/cuTensorNet) — coherent noise simulation
    GPU #2: CUDA-QX decoder (this module)      — syndrome decoding

The key insight: CUDA-QX decoders operate on parity check matrices and
syndrome vectors, not circuits. We extract the parity check matrix from
Stim's detector error model and feed Maestro's raw measurement output
(converted to syndromes) into the GPU decoder.

Usage:
    from cudaqx_decoder_bridge import CUDAQXDecoder

    # Initialize with Stim circuit (for detector error model)
    decoder = CUDAQXDecoder(stim_circuit, decoder_type='nv_qldpc_decoder')

    # Decode Maestro's raw measurements
    lep, lep_std, decode_time = decoder.decode_raw_measurements(raw_measurements)

Requirements:
    pip install cudaq cudaq-qec   (Linux x86_64 + CUDA 12+)
"""

import time
import numpy as np

try:
    import cudaq_qec as qec
    CUDAQX_AVAILABLE = True
except ImportError:
    CUDAQX_AVAILABLE = False
    print("Cuda QX decoder not available.")


def check_cudaqx():
    """Check if CUDA-QX is available and raise informative error if not."""
    if not CUDAQX_AVAILABLE:
        raise ImportError(
            "CUDA-QX QEC library not found.\n"
            "Install with: pip install cudaq cudaq-qec\n"
            "Requires: Linux x86_64, CUDA 12+, NVIDIA GPU\n"
            "Docs: https://nvidia.github.io/cudaqx/"
        )


class CUDAQXDecoder:
    """
    Bridge between Maestro's QEC measurement output and CUDA-QX GPU decoders.

    Extracts the parity check matrix from a Stim circuit's detector error
    model, initializes a CUDA-QX GPU decoder, and provides methods to
    decode raw measurement arrays from Maestro simulations.

    Supported decoders:
        - 'nv_qldpc_decoder': NVIDIA QLDPC BP-OSD decoder (29-35× GPU speedup)
        - 'single_error_lut': Single-error look-up table (baseline)

    Parameters:
        stim_circuit: A Stim circuit with detector/observable annotations
        decoder_type: Name of the CUDA-QX decoder to use
        **decoder_opts: Additional options passed to qec.get_decoder()
    """

    def __init__(self, stim_circuit, decoder_type='nv_qldpc_decoder',
                 **decoder_opts):
        check_cudaqx()

        self.stim_circuit = stim_circuit
        self.decoder_type = decoder_type
        self.decoder_opts = decoder_opts

        # Extract detector error model and build parity check matrix
        self._build_from_stim(stim_circuit)

        # Initialize CUDA-QX GPU decoder
        self._init_decoder()

    def _build_from_stim(self, stim_circuit):
        """
        Extract parity check matrix from Stim's detector error model.

        Stim's DEM describes which detectors are triggered by which errors.
        We convert this to a parity check matrix H that CUDA-QX expects.
        """
        dem = stim_circuit.detector_error_model()

        # Count detectors and observables
        self.n_detectors = stim_circuit.num_detectors
        self.n_observables = stim_circuit.num_observables

        # Build the parity check matrix from the DEM
        # Each DEM instruction "error(p) Dn Dm ... L0 L1 ..." tells us
        # which detectors an error mechanism triggers
        error_mechanisms = []
        self.error_probs = []
        self.observable_masks = []

        for instruction in dem.flattened():
            if instruction.type == 'error':
                prob = instruction.args_copy()[0]
                det_indices = []
                obs_indices = []
                for target in instruction.targets_copy():
                    if target.is_relative_detector_id():
                        det_indices.append(target.val)
                    elif target.is_logical_observable_id():
                        obs_indices.append(target.val)

                if det_indices:  # Only include if it triggers detectors
                    error_mechanisms.append(det_indices)
                    self.error_probs.append(prob)

                    obs_mask = np.zeros(max(self.n_observables, 1),
                                        dtype=np.uint8)
                    for o in obs_indices:
                        if o < len(obs_mask):
                            obs_mask[o] = 1
                    self.observable_masks.append(obs_mask)

        # Build H matrix: rows = detectors, columns = error mechanisms
        n_errors = len(error_mechanisms)
        self.pcm = np.zeros((self.n_detectors, n_errors), dtype=np.uint8)
        for col, dets in enumerate(error_mechanisms):
            for d in dets:
                if d < self.n_detectors:
                    self.pcm[d, col] = 1

        self.error_probs = np.array(self.error_probs)
        self.observable_masks = np.array(self.observable_masks)

        # Also compile Stim's m2d converter for extracting detector events
        self.m2d_converter = stim_circuit.compile_m2d_converter()

    def _init_decoder(self):
        """Initialize the CUDA-QX GPU decoder with our parity check matrix."""
        self.decoder = qec.get_decoder(
            self.decoder_type, self.pcm, **self.decoder_opts
        )

    def raw_to_syndromes(self, raw_measurements):
        """
        Convert Maestro's raw measurement bitstrings to detector events.

        Uses Stim's measurement-to-detection-event converter, which
        applies the detector definitions from the circuit to map raw
        measurement outcomes to syndrome bits.

        Args:
            raw_measurements: (n_shots, n_measurements) uint8 array

        Returns:
            (detector_events, actual_observables) tuple of arrays
        """
        expected_meas = self.stim_circuit.num_measurements

        # Handle measurement count mismatch (padding/truncation)
        if raw_measurements.shape[1] < expected_meas:
            pad = np.zeros(
                (raw_measurements.shape[0],
                 expected_meas - raw_measurements.shape[1]),
                dtype=np.uint8
            )
            raw_measurements = np.hstack([raw_measurements, pad])
        elif raw_measurements.shape[1] > expected_meas:
            raw_measurements = raw_measurements[:, :expected_meas]

        det_data = self.m2d_converter.convert(
            measurements=raw_measurements.astype(bool),
            separate_observables=True,
        )

        detector_events = np.array(det_data[0], dtype=np.uint8)
        actual_observables = np.array(det_data[1], dtype=np.uint8)

        return detector_events, actual_observables

    def decode_syndromes(self, detector_events):
        """
        Decode detector events using the CUDA-QX GPU decoder.

        Each row of detector_events is a syndrome vector. The decoder
        predicts which error mechanism(s) occurred, from which we infer
        the logical observable flips.

        Args:
            detector_events: (n_shots, n_detectors) uint8 array

        Returns:
            predicted_observables: (n_shots, n_observables) uint8 array
            decode_time: float, total decode time in seconds
        """
        n_shots = detector_events.shape[0]
        predicted_observables = np.zeros(
            (n_shots, max(self.n_observables, 1)), dtype=np.uint8
        )

        t0 = time.time()
        for i in range(n_shots):
            syndrome = detector_events[i]
            result = self.decoder.decode(syndrome)
            prediction = np.array(result.result, dtype=np.float64)

            # Convert soft information to hard decisions
            hard_prediction = (prediction > 0.5).astype(np.uint8)

            # Map error predictions to observable flips
            if len(hard_prediction) == len(self.observable_masks):
                for j, (err, obs_mask) in enumerate(
                    zip(hard_prediction, self.observable_masks)
                ):
                    if err:
                        predicted_observables[i] ^= obs_mask

        decode_time = time.time() - t0
        return predicted_observables, decode_time

    def decode_raw_measurements(self, raw_measurements):
        """
        Full pipeline: raw Maestro measurements → logical error rate.

        This is the main entry point. Takes raw measurement bitstrings from
        Maestro's MPS simulation and returns the logical error probability,
        using CUDA-QX's GPU decoder.

        Args:
            raw_measurements: (n_shots, n_measurements) uint8 array from
                              Maestro's counts_to_bitarray()

        Returns:
            (lep, lep_std, decode_time) tuple:
                lep: logical error probability
                lep_std: standard deviation of LEP estimate
                decode_time: GPU decode time in seconds
        """
        # Step 1: Convert raw measurements to detector events
        detector_events, actual_observables = self.raw_to_syndromes(
            raw_measurements
        )

        # Step 2: Decode using CUDA-QX GPU decoder
        predicted_observables, decode_time = self.decode_syndromes(
            detector_events
        )

        # Step 3: Count logical errors
        n_shots = detector_events.shape[0]
        fails = np.sum(
            np.any(predicted_observables != actual_observables, axis=1)
        )
        lep = fails / n_shots
        lep_std = np.sqrt(lep * (1 - lep) / n_shots) if n_shots > 0 else 0

        return lep, lep_std, decode_time


class PyMatchingBaseline:
    """
    PyMatching decoder wrapper with the same interface as CUDAQXDecoder.
    
    Provides a consistent API for comparison benchmarks between
    PyMatching (CPU) and CUDA-QX (GPU) decoders.
    """

    def __init__(self, stim_circuit, pymatching_decoder):
        self.stim_circuit = stim_circuit
        self.decoder = pymatching_decoder
        self.m2d_converter = stim_circuit.compile_m2d_converter()
        self.n_observables = stim_circuit.num_observables

    def decode_raw_measurements(self, raw_measurements):
        """Same interface as CUDAQXDecoder.decode_raw_measurements()."""
        expected_meas = self.stim_circuit.num_measurements

        if raw_measurements.shape[1] < expected_meas:
            pad = np.zeros(
                (raw_measurements.shape[0],
                 expected_meas - raw_measurements.shape[1]),
                dtype=np.uint8
            )
            raw_measurements = np.hstack([raw_measurements, pad])
        elif raw_measurements.shape[1] > expected_meas:
            raw_measurements = raw_measurements[:, :expected_meas]

        det_data = self.m2d_converter.convert(
            measurements=raw_measurements.astype(bool),
            separate_observables=True,
        )
        detector_events = np.array(det_data[0], dtype=np.uint8)
        actual_observables = np.array(det_data[1], dtype=np.uint8)

        t0 = time.time()
        predicted_observables = self.decoder.decode_batch_to_logical_flip(
            detector_events
        )
        decode_time = time.time() - t0

        n_shots = detector_events.shape[0]
        fails = np.sum(
            np.any(predicted_observables != actual_observables, axis=1)
        )
        lep = fails / n_shots
        lep_std = np.sqrt(lep * (1 - lep) / n_shots) if n_shots > 0 else 0

        return lep, lep_std, decode_time
