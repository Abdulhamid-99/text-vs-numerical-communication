# Replay the 64-byte retrieval comparison

From the repository root, use Python 3.13 with NumPy 2.3.5 and threadpoolctl 3.6.0:

```sh
python -m pip install numpy==2.3.5 threadpoolctl==3.6.0
python replay/verify.py
```

After installing those packages the second command runs offline on a CPU. It decodes all 989 saved numerical packets, scores each query against the 7,376 receiver documents in modified ArguAna, and rebuilds the text channel's dense ranks and reciprocal-rank fusion. It compares the resulting ten document indices for **each** query and channel against the original rankings, calculates NDCG@10 from relevance indices, reads text byte lengths from original frame headers and repeats the component bootstrap and paired sign-flip test. A missing input, stale checksum, mismatched ranking or incorrect paper result stops the check with a nonzero exit.

Expected ArguAna means are 0.64050765 for numerical messages and 0.61705984 for text. The numerical messages use 56 bytes each and text uses 63.3650 bytes on average. Across 122 query components the paired difference is 0.02344781, its 95% interval rounds to [0.0024, 0.0443], and the two-sided p-value rounds to 0.0302. Byte counts cover transmitted message frames. Setup and common model costs are separate in the paper.

## Inputs to the offline check

The compressed numerical evidence contains the original sign-packet bytes, the original text frame headers and packet digests, the receiver-side document vectors and decoded-text embeddings from the frozen encoder run, and the lexical ranks computed from the decoded text. Additional arrays contain the frozen tie order, positional relevance and exclusion indices, dependency groups, and reference top-ten indices. The payload of an extractive text message can reproduce benchmark text, so the package keeps its header and digest. The calculation uses stored embeddings and lexical ranks for the text channel. Reading `verify.py` alone does not run the encoder, fit a text scheme, or select a channel on development data.

The smaller [retrieval/replay](../retrieval/replay) check starts from ranks and supports a direct check against upstream judgments. The [retrieval experiment](../retrieval) runs the encoder. Use this directory to start from actual numerical packets and repeat the receiver's ranking work.

## Rebuild from original packet files

Install `tokenizers==0.22.0` in addition to the two packages above. Use the BAAI/bge-small-en-v1.5 `tokenizer.json` from snapshot `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The command checks its SHA-256. For a run created with the [ArguAna instructions](../retrieval), give the work, data, and tokenizer paths appropriate to your machine. For example:

```sh
python replay/from_packets.py arguana --run work/primary --data work/data --split work/data_split.json --tokenizer models/bge-small-en-v1.5/tokenizer.json
```

This route reads the actual selected text and numerical packet files, verifies the 384 sign bits against the sender's saved vectors, decodes every text payload and recomputes BM25 lexical ranks from the upstream corpus. It performs both receiver rankings and the same inference as the offline check. The text embeddings are the receiver's stored outputs for the decoded messages. Re-running their neural encoder is a separate operation in the experiment. Use `--output FILE.npz` to save a verified receiver package. The output must be a new file. Use `--compare replay/arguana_receiver.npz` to check each rebuilt array against the distributed evidence, including the SHA-256 digest of each actual text packet.

The same utility supports completed BRIGHT Biology and SciFact runs in the original archive layout. Pass the directory containing the per-query packet files and the corresponding prepared benchmark directory. The ArguAna NPZ distributed here contains only ArguAna results.

```sh
python replay/from_packets.py bright --run runs/bright_cached_len_v2 --data bright/prepared --tokenizer models/bge-small-en-v1.5/tokenizer.json
python replay/from_packets.py scifact --run runs/scifact_cached_len_v2 --data scifact/prepared --tokenizer models/bge-small-en-v1.5/tokenizer.json
```

BRIGHT has 103 queries and 57,359 documents. Its numerical and text means are 0.11298014 and 0.10174286. SciFact has 300 queries and 5,183 documents, with means of 0.66995828 and 0.72920010. These checks cover the two frozen primary channels and their descriptive transfer intervals. The paper's secondary channels and adaptive policies have separate supporting records.

The checker uses the original full-matrix float32 cosine products and dataset-specific BLAS thread counts. BLAS libraries and processors can handle near-ties differently. A difference in any saved top-ten row is reported as an error even if it leaves the rounded average unchanged.

## Attribution for the numerical evidence

`arguana_receiver.npz` contains results computed for this study from the BEIR release of ArguAna, distributed as adapted data under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The study filtered the receiver corpus, partitioned query components, encoded messages, and calculated receiver rankings. The file includes derived row indices in place of benchmark texts. The original model and datasets can be obtained from their maintainers.

- Henning Wachsmuth, Shahbaz Syed, and Benno Stein (2018). [Retrieval of the Best Counterargument without Prior Topic Knowledge](https://aclanthology.org/P18-1023/).
- Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych (2021). [BEIR](https://arxiv.org/abs/2104.08663). [BEIR ArguAna dataset and license](https://huggingface.co/datasets/BeIR/arguana).
- [BAAI/bge-small-en-v1.5 model and tokenizer](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a). Model weights and tokenizer assets are obtained separately.
