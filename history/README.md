# Development history

The project began as a pilot comparing text and numerical messages between language models. The first recorded implementation commit dates to 6 May 2026. The main branch contains the code and evidence used to reproduce the paper. Two archive branches preserve the development record that preceded it.

| Recorded date | Work | Evidence |
| --- | --- | --- |
| 6 May to 28 June 2026 | The pilot progressed from channel implementation and budget accounting to activation injection, control experiments and a task that required transmission. The last commit recorded a digest manifest of the June results. | [25 pilot commits](https://github.com/Abdulhamid-99/text-vs-numerical-communication/commits/archive/pilot-development) |
| 12 September 2026 | Two commits added the numerical relay, packet formats, tests and byte-preserving checkout settings. The state-transfer protocol records its freeze at 20:43 UTC that day. | [Two relay commits](https://github.com/Abdulhamid-99/text-vs-numerical-communication/commits/archive/relay-development) and [frozen protocol](../state_transfer/frozen/plan/protocol.json) |
| 13 to 15 October 2026 | The ArguAna protocol and development selection were recorded on 13 October. Held-out evaluation and the two transfer audits completed on 15 October. These dates come from the experiment records in the supplement. | [Supplement S3](../evidence/Supplement_S3.zip), specifically `arguana/selection_lock.json`, `arguana/test_execution_completed.json` and `audits/transfer_audit/` |
| 23 November 2026 | The reproducibility repository was prepared and tested. The archived development branches were connected to its history on the same date. | [Verification and run instructions](../README.md) |

## How the history was imported

Each exported commit keeps the original Git author, committer, author date and commit date, including the time zone. Each retained file has the same bytes and mode as in its source commit. Filtering the trees and parent links gives the archive commits new identifiers. The [commit map](commit-map.tsv) records the full original and archive identifiers.

The pilot export includes source code, configuration, tests, build files, two numerical layer-selection tables and the June digest manifest. It excludes raw pilot outputs and workspace notes. Six commit messages omit tool-attribution footers. Pilot results reflect the methods in use at the time. The reported paper results are documented on the main branch.

The September export contains only the changes introduced by the two author commits. Its source repository started from a separate automated baseline snapshot. That snapshot is listed in the map as excluded. The two author commits retain their own changes and dates.
