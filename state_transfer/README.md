# State transfer on private QA and navigation

The two agents use the same checkpoint within each model pair. Each task contains 64 paired families. The sender sees one world's private message and the recipient sees the same permitted context across the two worlds. The study measures generated UTF-8 text, ordered full-width int4 states and states projected onto a 512-component PCA basis. The four original task/model PCA bases were fitted on separate synthetic calibration inputs and are included under `frozen/plan/assets`.

## Check the frozen inputs

Run these commands from `state_transfer/`. The host controller uses Python 3.11 or newer. This first check needs only the standard library.

```sh
python -B state_study.py verify
```

This verifies every frozen Python source file, the four basis and descriptor hashes, paired input partitions, exclusion commitments and the original held-out manifest. The published protocol SHA-256 is `c130f965ebe9a14f95071200657df3789560a4c37120d6ea73c19ae806fd12bb`. The 50 files under `runtime/` are the complete source inventory committed to by that protocol. The synthetic files called `private` are private to one role of the experiment. The host retains them for scoring. The isolated model processes receive only their assigned inputs.

## Environment and checkpoints

Use Linux containers, Docker with NVIDIA GPU support, and a GPU/driver capable of CUDA 13.0 and bfloat16. The reference system had a 16 GB RTX 5080, a CUDA 13.0 build of PyTorch 2.11.0, Transformers 4.57.6 and bitsandbytes 0.49.2. Allow room for roughly 31 GB of checkpoints, the Docker image and the generated study evidence.

```sh
docker build --platform linux/amd64 -t paper-state-repro:cu130 .
```

The Dockerfile starts from a public, digest-pinned Python base. Its source and primary model dependencies match the original experiment. The frozen protocol also records the authors' original local image ID. A rebuilt image gets its own ID. For a new run, the controller writes both identities to `execution_binding.json` in the output directory and binds fresh copies of the input files to the image that actually runs. The scientific settings, model revisions, frozen source, PCA bases, private worlds and loss masks remain the same.

The model weights come directly from the following repositories. Obtain access to Llama before downloading the gated files. Install the Hugging Face CLI on the host and sign in there if needed. Use a dedicated `CACHE` directory for this study. Replace `CACHE` and `OUTPUT` in the examples with your own paths.

```sh
hf download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --cache-dir CACHE --include "*.json" "*.safetensors" "merges.txt"
hf download meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 --cache-dir CACHE --include "*.json" "*.safetensors"
python -B state_study.py verify-cache --cache CACHE --image paper-state-repro:cu130
```

The cache check reads and hashes the checkpoint shards, tokenizer and configuration files against the study inventory. Checkpoints are loaded offline by revision. Model notices and applicable upstream terms are in [model_notices](model_notices/). Built with Llama. Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

## One real sender-to-receiver check

```sh
python -B state_study.py smoke --cache CACHE --image paper-state-repro:cu130 --output SMOKE_OUTPUT
```

This uses Qwen on the first frozen QA family. The sender generates a message, re-encodes its ordered states and produces a PCA packet. A separate, offline receiver container sees its own private view and that packet. The host checks the packet digest, byte count and single span injection, then grades the answer against the hidden world. Read `smoke_result.json` for both the transport outcome and the task score. A completed transfer can yield an incorrect answer. The smoke validates small checkpoint files and checks that all Qwen shards are present. Use `verify-cache` for full weight hashes.

The original-image release check reproduced the recorded first QA case exactly. Qwen generated 39 UTF-8 bytes (27 tokens). The PCA packet contained 7,084 bytes and had SHA-256 `5036ab3ee552dfa4271425cb3e838562b192c9d0250f88cd9bd1a5012ec23ef7`. The receiver consumed the packet once and produced the correct 16-letter answer in 10 tokens.

## Full held-out experiment

```sh
python -B state_study.py run --cache CACHE --image paper-state-repro:cu130 --output OUTPUT
```

Use a new output path outside the checkout and model cache. The controller validates both model caches, runs eight isolated stages and then evaluates their sealed outputs in the same Linux image on CPU. This requests 512 sender and 3,072 receiver generations on the original held-out families. Outputs are written under `OUTPUT/study`, `OUTPUT/analysis` and `OUTPUT/analysis_statistics`. The complete run keeps per-example traces, captures and actual packets. An incomplete stage or a failed validation stops the analysis. Model/configuration or hardware differences can affect generated outputs.

To regrade an existing complete run without performing new model inference, use its saved protocol and image. The exact numeric checks validate reconstruction bytes, so the command runs the frozen analyzer in the producer's Linux container with the recorded CPU thread settings.

```sh
python -B state_study.py analyze --plan OUTPUT/plan/protocol.json --data OUTPUT/data --study OUTPUT/study --output OUTPUT/reanalysis
```

For the eight supplementary budget and fidelity tables, follow the [S2 analysis instructions](diagnostics/README.md).
