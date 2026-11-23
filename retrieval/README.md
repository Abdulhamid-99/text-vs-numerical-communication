# Remote semantic retrieval

Start in the repository root. Three datasets are used. ArguAna is an exploratory, modified BEIR experiment with a separate development set and a frozen internal test set. BRIGHT Biology and SciFact apply a fixed comparison to different collections.

## Recreate the primary comparison from the original public inputs

Use Python 3.13 and a fresh virtual environment. The study used PyTorch 2.10.0 with CUDA 12.8 on Windows. Install a compatible PyTorch wheel for your hardware, then run the commands below. The encoder also supports CPU inference. Use a path outside the checkout to keep downloaded data, model output and packet contents private.

```sh
python -m pip install -r retrieval/requirements.txt
python retrieval/arguana.py prepare --work ../arguana-work
python retrieval/arguana.py primary --work ../arguana-work
```

Preparation downloads pinned Hugging Face Parquet files and qrels from the BEIR authors, reconstructs the original JSONL, and verifies that all three input files match the original BEIR distribution byte for byte. The original ZIP works offline. Use `prepare --work ../arguana-work --archive /path/to/arguana.zip` to supply a local copy. Preparation verifies SHA-256 for each source file, constructs the content/debate component split, and checks both that split and the fitted IDF dictionary against their frozen identities. It excludes five queries with missing targets, partitions 412 development and 989 test queries, and keeps 7,376 documents available to the receiver. The raw data and corpus-derived dictionary are created locally.

The model loader gets revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` of `BAAI/bge-small-en-v1.5` and checks the model, configuration and tokenizer files against the original hashes in `protocol.json`. To run fully offline, supply the verified archive and set `BGE_MODEL_DIR` to a local copy of that checkpoint. Set `BGE_DEVICE=cpu` to force CPU encoding.

The primary run encodes the corpus and 989 original queries, generates every sign and text packet, counts the transmitted bytes including the eight-byte header, decodes each packet, and ranks documents using the receiver. It uses the two configurations already chosen on development. The numerical receiver uses cosine similarity on the decoded sign vector. The text sender transmits an extractive lead-and-keyword message; the receiver combines dense and BM25 rankings via reciprocal rank fusion. The sender has neither relevance labels nor the receiver's searchable corpus.

| Published primary result | Numerical | Text |
| --- | ---: | ---: |
| NDCG@10 | 0.640507651 | 0.617059842 |
| Mean transmitted bytes | 56.000000 | 63.365015 |
| Queries scored | 989 | 989 |

Fresh execution on the release machine regenerated all 1,978 primary packets and all 1,978 top-ten lists from the raw corpus, queries and public checkpoint. They matched the archived packets byte for byte and the ranked document indices element for element. The published 95% component-bootstrap interval and sign-flip result also matched when recomputed from these new rankings. This release check used Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cu128 and SciPy 1.18.1; the original recorded SciPy version was 1.16.1. The primary raw sign comparison does not fit a PCA basis.

The completed folder `primary/` contains per-query metrics, rankings, actual binary packets, query and corpus embeddings and a model-call ledger. A failed stage returns a nonzero exit status. Only a completed stage is renamed into its final destination. A rerun requires a fresh destination. Float32 encoder and BLAS differences across platforms can affect very close document ranks. Report observed deviations alongside the archived numbers.

## Repeat selection and evaluate the budgets

```sh
python retrieval/arguana.py development --work ../arguana-work
python retrieval/arguana.py heldout --work ../arguana-work --selection ../arguana-work/development/selection.json
```

Development searches 28 numerical encodings and 315 text configurations on the 412 development queries. It maximizes mean NDCG@10, breaking ties with mean bytes and then configuration key. Semantic-window selection invokes the encoder repeatedly for candidate messages and is substantially more expensive than the frozen primary comparison. The five nominal caps are 32, 64, 128, 256 and 512 bytes. Omitting `--selection` from the held-out command uses the archived development choices in `selection.json`. The runnable grid covers the candidate search and the nominal caps. The [secondary analyses](secondary/README.md) cover the budget envelope, fitted codec and routing policy reported in Supplement S3.

To compute the published uncertainty estimates using outputs of your own primary run, see [the fresh-output workflow](replay/README.md). Numerical equality at one budget does not verify a rerun of the complete development grid.

## Other checks

- `python retrieval/test_scoring.py` tests missing rows, missing or inconsistent relevance labels, valid zero scores and invalid budget selections.
- [Primary statistical replay](replay/README.md) recomputes the paired difference, 95% component bootstrap interval, two-sided component sign-flip test and byte saving from recorded ranks and packet lengths. This takes no GPU or model downloads.
- [BRIGHT Biology and SciFact](transfer/README.md) documents data preparation and the two fixed transfer comparisons.
- The public data source and recorded checkpoint identifiers are in [protocol.json](protocol.json). Data and model weights are obtained from their upstream providers.
