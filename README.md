# Text and Numerical Representations Under Shared Communication Budgets

**Abdulhamid Bakr Haras, Abdullah Alshanqiti, El-Tayeb k. Karrar and Emad Nabil**  
Department of Computer Science, Islamic University of Madinah, Saudi Arabia

This repository accompanies our study of what two agents can communicate under a shared message budget. In the first experiment, a sender observes private information and sends generated text or a packet of model states to a second language model. In the second, a remote receiver uses a text message or a quantized query vector to rank documents. These tasks have different receivers and are evaluated separately.

The [development history](history/README.md) traces the May 2026 pilot through the September state-transfer work, the October retrieval experiments and the November reproducibility release.

## Reproduce the results

The [experiment records release](https://github.com/Abdulhamid-99/text-vs-numerical-communication/releases/tag/experiment-records-v1) contains all 2,816 nonempty state-transfer packets and two extracts from the ArguAna model-call logs. [Download the ZIP](https://github.com/Abdulhamid-99/text-vs-numerical-communication/releases/download/experiment-records-v1/experiment-records.zip) to check the original packet bytes with the [offline replay](state_replay/README.md).

For the fastest independent check, create a Python 3.13 environment and install the packages below. The verification itself runs offline on a CPU.

```sh
python -m pip install -r requirements-replay.txt
python verify.py
```

The same checks can run in a Linux container. Build the image once, then run it with networking disabled:

```sh
docker build --platform linux/amd64 -f Dockerfile.replay -t paper-cpu-replay .
docker run --rm --network none paper-cpu-replay
```

The state-transfer results cover 64 paired families for each of two tasks, tested with Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Six of the 16 prespecified comparisons favored text after correction. On 989 held-out ArguAna queries, under a 64-byte ceiling, the selected text channel reached 0.6171 NDCG@10. The 56-byte sign packet reached 0.6405 and used 11.62% fewer message bytes on average. SciFact favored text, and the BRIGHT Biology interval included zero.

| To reproduce | Instructions |
| --- | --- |
| State-transfer messages and answers using the two language models | [Docker, frozen inputs and model runs](state_transfer/README.md) |
| State-transfer scores from the recorded receiver answers and delivery decisions | [Offline state-transfer replay](state_replay/README.md) |
| ArguAna messages, byte counts and rankings from the benchmark and the encoder | [Retrieval experiment](retrieval/README.md) |
| Fixed 64-byte comparisons on BRIGHT Biology and SciFact | [Transfer experiments](retrieval/transfer/README.md) |
| ArguAna rankings from saved receiver inputs and sign packets | [Packet and ranking replay](replay/README.md) |
| Measured message budgets, fitted codec overhead and routing results | [Secondary retrieval comparisons](retrieval/secondary/README.md) |
| Paired uncertainty, byte savings and the journal tables | [ArguAna comparison](retrieval/replay/README.md) and [supplement checks](evidence/README.md) |

## Verification for this version

On 23 November 2026, the fixed 64-byte comparison was run again from the original benchmark inputs and the pinned BGE checkpoint. The new runs covered 989 ArguAna queries, 103 BRIGHT Biology queries and 300 SciFact queries. Across the two channels, all 2,784 transmitted packets and all 2,784 top-ten rankings matched the archived outputs. The 343-candidate ArguAna development search was not repeated in this check.

For state transfer, the CPU replay rescored all 3,072 saved receiver answers and reproduced the 180 budget-sweep cells. One Qwen QA example was also run with separate sender and receiver containers, first with the archived image and again with the image rebuilt from the Dockerfile in this repository. Both produced the same 39 bytes of sender text, the same 7,084-byte PCA packet and the same correct receiver answer as the original run. The full 3,584 model generations were not repeated for this version.

## Experiment inputs

The language-model experiment uses synthetic questions and maps. The retrieval experiment uses the original benchmark distributions. Follow the study-specific instructions to obtain the pinned inputs. Model weights and benchmark text come from the upstream providers under their own terms.

| Input | Upstream source |
| --- | --- |
| Qwen2.5-7B-Instruct | [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Llama-3.1-8B-Instruct | [Meta model card and access terms](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| BGE-small-en-v1.5 | [BAAI model card](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| ArguAna | [BEIR distribution](https://huggingface.co/datasets/BeIR/arguana) |
| BRIGHT Biology | [BRIGHT benchmark](https://huggingface.co/datasets/xlangai/BRIGHT) |
| SciFact | [BEIR distribution](https://huggingface.co/datasets/BeIR/scifact) |

For archived comparisons, use the precision and numerical-library settings in the study instructions. Ranking scores close to a tie can change order across BLAS implementations. An interrupted evaluation is an incomplete run and must be resumed or reported separately.
