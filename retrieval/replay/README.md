# ArguAna query-level results

Run the 64-byte held-out comparison from the repository root with Python and NumPy. The reference environment used NumPy 2.3.5.

```sh
python retrieval/replay/check.py
```

The checker reads 989 paired rows, scores NDCG@10 for the two frozen channels, measures mean transmitted bytes and computes the confidence interval and two-sided p-value. It checks the result against the paper's full-precision values and returns a nonzero exit code on a discrepancy.

| Measure | Recomputed from the archived rankings and packet lengths |
| --- | ---: |
| NDCG@10, 384-dimensional sign packet | 0.64050765 |
| NDCG@10, extractive text with hybrid ranking | 0.61705984 |
| Mean bytes per sign / text message | 56.0000 / 63.3650 |
| Relative saving in message bytes | 11.6232% |
| Paired NDCG@10 difference | 0.02344781 |
| 95% component-bootstrap interval | [0.00244464, 0.04426330] |
| Two-sided component sign-flip p | 0.03023970 |

## Checking the rankings against the dataset

Download and extract the [BEIR ArguAna release](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip). Specify the directory containing `corpus.jsonl`, `queries.jsonl` and `qrels/test.tsv`. A flat `test.tsv` is also supported. The checker validates the hashes of all three source files and scores each of the 1,978 rankings against the positive document in the original qrels.

```sh
python retrieval/replay/check.py --beir-dir /path/to/arguana
```

## What each observation contains

`observations.csv` records the selected numerical and textual channels at the same 64-byte ceiling. Query and document indices are zero-based line numbers in BEIR's original JSONL files. The ordered top ten for each channel are stored as space-separated document indices. Each held-out query has one relevant document. The recorded hit gives its rank from 1 to 10, or 0 for a miss. The two byte counts come from the lengths of the transmitted binary files, including packet headers. They exclude model, index and setup costs.

The 122 integer component labels preserve the lexical ordering of the original component hashes. The resampling unit is a connected component. For the confidence interval the code draws 20,000 bootstrap replicates; each replicate divides the sum of sampled differences by its sampled query count. For the two-sided test it draws 100,000 component sign assignments. Seeds, tie tolerance and the p-value correction match the frozen analysis.

This is a scoring replay. It requires no network access or model inference. For an independent run that generates the messages and rankings from source inputs, use the [retrieval experiment instructions](../README.md).

## Recomputing the interval and p-value from your own run

After completing the primary ArguAna experiment, use its working directory to
export the rankings and lengths of your generated packets. The export verifies
the frozen split, input hashes and query order and scores both top-ten rankings
against the public relevance labels. Use a new output filename.

```sh
python retrieval/replay/export.py --work runs/arguana --out observations.new.csv
python retrieval/replay/check.py --observations observations.new.csv --beir-dir runs/arguana/data
```

The second command computes the interval, two-sided p-value and byte saving
from your output and exits with a discrepancy if it differs from the published figures.

## Rebuilding the outcome table from the historical run

`export.py` reads a local copy of the original run directory. It verifies the identities of the selected channels, split, held-out query order and both ranking arrays, maps receiver indices back to corpus row numbers, and measures all 1,978 packet files. It writes only derived ranking indices, group numbers, hit ranks and byte counts. No text, recoverable packets, embeddings, or model weights are distributed here.

```sh
python retrieval/replay/export.py --historical-run /path/to/run_merged_cached_len_v2 --beir-dir /path/to/arguana --out observations.new.csv
python retrieval/replay/check.py --observations observations.new.csv --beir-dir /path/to/arguana
```

The archived CSV has SHA256 `c99c003ea5a6905404479cd2f4ccf4973ecee70b2204c2ac775a9c66c15af428`; its line endings are LF.

## Actual wrong-message control

After `arguana.py primary` has completed, rerank each of the 989 recipients using a different component's full sender vector and its actual 56-byte sign packet. Run from the repository root.

```sh
python retrieval/replay/wrong_message.py --work runs/arguana --stage primary
```

After `arguana.py heldout`, use `--stage heldout` to check the other two selected numerical channels as well. The checker verifies the complete donor permutation from seed 91827 against the original pre-outcome digest. It decodes the donor's packet bytes, ranks the receiver's 7,376 documents and scores the resulting top ten against the original **recipient's** positive document. The reference result is zero Hit@1, MRR@10 and NDCG@10 in all four conditions (full vector, raw sign, raw four-bit and 32-component four-bit PCA).

For the corresponding positive control, the checker restores original query order, ranks the same decoded numerical messages again under the primary scoring settings and compares mean NDCG@10 with that run's recorded matched result. It also prints the donor-aligned score within the shuffled ranking as a diagnostic. Near ties in float32 can change rank order between matrix layouts. The recipient score is the wrong-message outcome.

For the historical merged output, use `--historical-run /path/to/run_merged_cached_len_v2 --beir-dir /path/to/arguana --stage heldout` in place of `--work`.
