# BRIGHT Biology and SciFact transfer

The 64-byte comparison transfers two methods chosen on ArguAna development to every test query in two other datasets. No thresholds, text settings or encoders are fitted on the transfer queries. The numerical packet carries 384 signs. The text sender uses the frozen `lead_keywords` rule and the ArguAna document-frequency dictionary. The receiver decodes the transmitted token IDs and combines dense retrieval with BM25 using reciprocal rank fusion.

Run these commands from the repository root after installing the [retrieval environment](../README.md). Parquet input requires `pyarrow==26.0.0`. If you already prepared ArguAna, skip that command and point the IDF argument to the existing work directory. Both runs write actual sender packets, per-query byte costs, top-ten rankings and metrics. A failed run exits with an error. Choose a fresh output directory for each attempt.

```shell
python -m pip install pyarrow==26.0.0
python retrieval/arguana.py prepare --work ../arguana-work
python retrieval/transfer/run.py prepare bright --data ../transfer-data
python retrieval/transfer/run.py prepare scifact --data ../transfer-data
python retrieval/transfer/run.py run bright --data ../transfer-data --idf ../arguana-work/idf_final.json --out ../bright-fixed
python retrieval/transfer/run.py run scifact --data ../transfer-data --idf ../arguana-work/idf_final.json --out ../scifact-fixed
```

The download step checks SHA-256 digests before reading the files. BRIGHT uses the original Biology queries and documents from [xlangai/BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT/tree/3066d29c9651a576c8aba4832d249807b181ecae). SciFact uses [BEIR's corpus and queries](https://huggingface.co/datasets/BeIR/scifact/tree/b3b5335604bf5ee3c4447671af975ea25143d4f5) and [the BEIR test relevance labels](https://huggingface.co/datasets/BeIR/scifact-qrels/tree/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029). You can also supply the [original BEIR archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip) as `../transfer-data/scifact.zip`. Both SciFact sources give the same query text, document text, labels and ordering.

| Dataset | Queries | Documents | Components | Numerical NDCG@10 | Text NDCG@10 | Mean numerical bytes | Mean text bytes | 95% interval for numerical minus text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BRIGHT Biology | 103 | 57,359 | 100 | 0.1130 | 0.1017 | 56.00 | 62.12 | [-0.01949, 0.04338] |
| SciFact | 300 | 5,183 | 247 | 0.6700 | 0.7292 | 56.00 | 46.47 | [-0.09086, -0.02640] |

These are the paper's reference values. The descriptive intervals use 20,000 paired component-bootstrap replicates with seed 1357913. Numerical ranking uses the decoded sign message. Text ranking uses only the decoded extractive message and the receiver's own corpus. Reference BLAS thread counts are 6 for BRIGHT and 20 for SciFact. The script checks the runtime inputs and exports its computed results even when the numerical values differ on other hardware.

This entry point covers the two prior-selected channels. The broader text grid, alternative codecs, routing and shuffled conditions are documented in Supplement S3.
