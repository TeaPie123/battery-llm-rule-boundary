# Final revision code review

Review date: 2026-09-27

## Scope

The review covered all 22 files collected from the final revision experiment workspace, the three reused historical components, the public synthetic datasets, aggregate result tables, and the final six-page paper.

## Checks completed

- parsed/compiled every revision Python file;
- compared all five strict `>` thresholds and state-applicability rules across dataset construction, baselines, BoundarySet, RuleVerifier, and validity analysis;
- verified that final adjacent differences reset within each cycle;
- verified that split builders and audits enforce record-ID and cycle disjointness;
- checked the LoRA sample-weighted loss against the formula in the paper;
- checked candidate-token scoring and the 0.5 decision threshold;
- checked validation-only threshold selection for Isolation Forest, Random Forest, and LSTM;
- checked exact Clopper-Pearson intervals, per-record McNemar pairing, and the three-test Holm family;
- matched all displayed Table III metrics to `cross_method_comparison.csv`;
- matched Tables II, IV, and VI and the data-validity statements to released artifacts;
- checked for credentials, private keys, large model/data files, and local absolute paths before publication.

## Findings and disposition

### Fixed in the GitHub release

1. **Holm correction was not previously emitted by the final comparison script.** The raw exact McNemar p values were correct and the paper’s adjusted values were also correct, but the adjustment step was not part of the saved code path. `revision_build_baseline_comparison_v2.py` now writes `prespecified_boundary_lora_mcnemar_holm.csv`, and `verify_paper_results.py` independently recomputes it.
2. **Cycle duplicate checking normalized keys after the check.** This could miss a duplicated integer-valued cycle identifier in a hand-edited design file. The release normalizes the identifier before checking. It does not change the recorded split or results.
3. **The provenance builder referred to the filenames used on the Linux run server.** The public release stores those files under `scripts/revision/`; its path list now matches the published layout. Hash semantics and experiment outputs are unchanged.

### Documented, not result-invalidating

1. `design_grouped_split.py`, `revision_build_baseline_comparison.py`, and `smoke_test.py` are superseded/non-final utilities. They remain for completeness but are not final-paper entry points.
2. The historical `scripts/new_error_linux_full_total_confirmed.py` uses a global previous row and contains old prompt wording with a 0.02 V cell threshold, even though its executable constant is 0.05 V. It must not be used to regenerate final labels. The final builder ignores the stored old label and recomputes the 0.05 V rule within cycle.
3. `scripts/23_build_boundary_train_v4.py` also tries to build the old v4 merged dataset after generating boundary samples. The final pipeline consumes only its public `boundary_train.jsonl` output through `build_revision_boundary_comparison.py`.
4. The grouped RuleVerifier evaluator includes an old-result regression guard whose record-level inputs are private and therefore absent from the public repository. Full reruns require those inputs; released aggregate outputs remain checkable.
5. GPU-dependent training and full inference were not rerun during this packaging review because the private data, adapters, and local model are intentionally excluded. Syntax, static logic, stored prediction/metric audits, and public result checks passed.
6. Table V contains two presentation-rounded numbers rather than exact six-decimal roundings: record 900056 Base-LoRA is stored as `0.2942149639` (paper: `0.294200`) and Boundary weighted is `0.0330859795` (paper: `0.033090`). Predictions and every aggregate metric are unchanged. Exact values are published in `results/paper_final/representative_boundary_cases.csv`.

## Verdict

No reviewed issue invalidates the final headline results. The final code/result chain is internally consistent for the released artifacts. Scientific limitations remain those stated in the paper: one data source, one compact backbone, one seed, limited natural near-threshold coverage, duplicated voltage fields, and a synthetic fixed BoundarySet.
