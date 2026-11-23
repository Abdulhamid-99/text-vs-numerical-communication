# State-transfer results from saved model answers

From the repository root, run the following command with Python, NumPy and SciPy. The release check used Python 3.13.5, NumPy 2.3.5 and SciPy 1.18.1. This step runs on CPU with the bundled synthetic families and saved answers.

```sh
python state_replay/replay.py
```

The replay applies the frozen scorer to all 3,072 receiver answers, including the planner for navigation and evaluation in the original and alternate worlds. It reconstructs 4,608 primary delivery decisions from the saved generated-message and no-message outcomes. Exactly 16 paired comparisons enter the global Holm correction, of which six favor text. A separate sweep reconstructs all 11,520 zero-loss delivery decisions and compares all 180 aggregate rows with Supplement S2. Incomplete observations, unexpected family cells and invalid packet costs cause the command to fail.

`observations.zip` contains the original held-out evaluation status, receiver-level answers and task scores, policy records, and the reference statistics. The first three feed the replay. The reference statistics are used only for comparison after scoring. Every field in the statistics is checked, with a tolerance of 1e-12 for floating-point library variation. The original statistics digest is also recorded in Supplement S1's provenance. Synthetic task inputs and the scorer are shared with [the state-transfer experiment](../state_transfer/).

The included packet index records the measured byte length and SHA-256 of each of the 2,816 nonempty receiver-input packets. Each length was checked against the file and its original input descriptor during packaging. The CPU replay matches these lengths to the observed wire-bit counts used in the comparisons. With the **original archived packet files** present, the following command also reads the packet bytes, hashes them and checks all 2,816 lengths directly. Supply the directory containing the model/channel input partitions.

```sh
python state_replay/replay.py --packet-root /path/to/heldout_v3
```

The default run reports zero serialized packets rehashed. Both modes start with saved model answers. For fresh inference, see [the state-transfer experiment](../state_transfer/).
