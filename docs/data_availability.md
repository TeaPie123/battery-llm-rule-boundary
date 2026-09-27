# Data and artifact availability

## Public in this repository

- all final revision source scripts;
- the 71-case synthetic BoundarySet;
- the 1,017-row synthetic boundary-training file;
- aggregate metrics for all eight compared methods;
- synthetic per-case predictions for the three LoRA variants;
- natural near-threshold aggregate results;
- RuleVerifier correction and timing summaries;
- aggregate data-validity and sensitivity tables.

## Not public

- the original battery Excel file and processed 614,497-record array;
- record-level natural train/validation/test JSONL files;
- record-level natural prediction outputs;
- pretrained Qwen files;
- LoRA adapters, LSTM checkpoints, and serialized conventional models;
- intermediate feature and sequence archives.

These exclusions prevent redistribution of source records and keep the repository within normal GitHub size limits. The code expects private inputs to be restored under the paths documented in `docs/reproduce.md`.

## Integrity approach

The public aggregate CSV files are sufficient to recompute every displayed metric in Tables II-IV and VI. Table V uses only the published synthetic cases. Run `python scripts/revision/verify_paper_results.py` to check the released artifacts.
