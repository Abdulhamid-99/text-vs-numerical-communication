# Supplement S2 from a completed state-transfer run

This CPU analysis builds the eight CSV tables in Supplement S2. Run it from the `state_transfer` directory after the eight sender/receiver stages and the primary analysis have completed. The command reads the frozen plan and its four bases, the observed sender captures, the actual receiver records, and the scored family-level analysis. The model weights are unnecessary for this step.

```sh
python -B diagnostics/reproduce_s2.py --plan OUTPUT/plan/protocol.json --run OUTPUT/study --analysis OUTPUT/analysis --output OUTPUT/s2
```

Use a fresh output directory. The script checks completion, plan bindings and file digests before publishing the output. A changed packet, missing run stage or failed check stops the command. NumPy and SciPy are the only external Python dependencies. The release check used Python 3.13, NumPy 2.3.5 and SciPy 1.18.1 on CPU.

The `calibration/` directory contains the original artifact manifests, configuration, synthetic calibration traces and singular spectra for Qwen and Llama. The code verifies the files it consumes against the corresponding frozen manifests. Each manifest also identifies other files from the original calibration run. Those files are unnecessary for the exported tables.

The budget sweep reuses the observed delivered or no-message response under each of the 15 caps. The held-out fidelity check reconstructs the 256 generated-original spans using an independent packet decoder. The per-family errors use float64 matrix multiplication followed by float32 reconstruction. A separate einsum calculation checks numerical sensitivity and reconstruction hashes. The output receipt records hashes of all eight tables and the number of reconstructed tensors that match the receiver.

When run against the recorded experiment, six tables matched the journal supplement byte for byte. The other two, `primary_contrasts.csv` and `high_budget_contrasts.csv`, matched in every field except for confidence interval rounding, with differences of at most 2.14e-14 percentage points. The primary reconstruction matched 255 receiver hashes. At least one of the two summation methods matched all 256. Fresh neural runs should be compared using the measured results from those runs.
