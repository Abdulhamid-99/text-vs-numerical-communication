# Secondary retrieval analyses

From the repository root, run the following command with Python 3.11 or newer and NumPy installed. It runs offline.

```sh
python retrieval/secondary/verify.py
```

**Feasible-budget envelope.** The command reapplies the original maximization and tie-breaking rule to the 315 text and 28 numerical development candidates. It compares the selected keys with the pre-test protocol, then recomputes the held-out means, wins, losses and ties on all 989 queries for five ceilings. Only the 256-byte text choice differs from the nominal selection. The retained 128-byte text configuration scores 0.6469 at 127.42 mean bytes. Its numerical comparator scores 0.6551 at 204 bytes. The score replay uses the per-query metrics in Supplement S3. Wire costs for this envelope are the recorded packet-bank mean and maximum for each configuration.

**Fixed switch.** For each query, the rule chooses complete text when its shortest serialized form is no larger than the numerical packet. The replay applies measured lengths to the evaluated branch outcomes, checks every reconstructed row against the archived policy, and recomputes the SciFact paired component interval. In ArguAna all 989 queries use numerical messages. In BRIGHT Biology three queries use complete text and 100 use numerical messages. In SciFact the counts are 222 and 78, respectively. The policy reaches NDCG@10 of 0.7057 at 45.10 mean bytes on SciFact. The matched-runtime ArguAna policy score equals the always-numerical score. These are outcome and message-cost comparisons. Neither branch's online inference time is estimated.

**Fitted binary codec.** The original four development results select the raw two-centroid codec. The check decodes all 1,392 saved packets with the same fitted asset and compares the resulting vectors, byte for byte, to the receiver outputs. The 56-byte packets share a 3,080-byte codebook. The amortized cost adds the codebook once per evaluated session. Averaging the archived per-query retrieval outcomes gives NDCG@10 of 0.5684 on ArguAna, 0.0596 on BRIGHT and 0.6356 on SciFact. This check reconstructs the decoder and the accounting. It uses recorded relevance scores and does not rerun fitted-codec ranking or neural inference.

## Files

`dev_results.json` is the original development scoreboard identified by the SHA-256 commitment in the frozen budget-envelope protocol. `codec_development.json` contains the four secondary development scores. `binary_codec.py` is the original two-level training and packet codec implementation. `two_means.bin` is its selected training-fitted asset. `codec_observations.npz` holds the original fitted-codec packets in evaluation order and supplementary ArguAna and SciFact score rows. BRIGHT score rows and the main retrieval records are read directly from [Supplement S3](../../evidence/README.md). `routing_inputs.npz` stores integer message lengths and SciFact component indices. The branch inputs and row-order conventions are described in [routing.py](routing.py).

Derived retrieval measurements use [BEIR ArguAna](https://huggingface.co/datasets/BeIR/arguana), [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) and [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT). The distributed numerical measurements derived from the BEIR datasets retain [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) attribution and those derived from BRIGHT retain [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Row-order packet, size and metric arrays contain no benchmark texts or model weights.
