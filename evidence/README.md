# Journal supplements

These ZIP files are the three supplements prepared for *Text and Numerical Representations Under Shared Communication Budgets*. Their tables, recorded results, figures and checkers match the corresponding journal files byte for byte.

From the repository root, run the following command with Python and NumPy installed. The checks were run with Python 3.13.5 and NumPy 2.3.5.

```sh
python evidence/verify.py
```

The command checks the SHA256 digest of each ZIP, extracts it to a fresh temporary directory and runs the checker included in the supplement. It exits with an error if a file is missing, a digest differs or a checker fails. It can run offline on a CPU.

| Supplement | What the command verifies |
| --- | --- |
| S1 | Recomputes the 16 prespecified paired comparisons from their counts, including exact McNemar tests and the global Holm correction. Six corrected comparisons favor text. |
| S2 | Checks the saved budget sweep (180 rows), family-level PCA diagnostics (256 rows) and packet-cost arithmetic. The intervals and fidelity measurements are stored results. |
| S3 | Verifies 123 file hashes and recomputes mean retrieval scores for 103 conditions using the saved per-query metrics from ArguAna, BRIGHT Biology and SciFact. |

These checks use saved experimental outputs. Model inference, decoding transmitted packets, and retrieving and ranking documents require the experiment or replay commands elsewhere in this repository. Each ZIP includes a README describing its inputs and scope.
