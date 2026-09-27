# Revision script inventory

## Final pipeline

- `audit_split_leakage.py`: quantify leakage in the historical split.
- `audit_rule_consistency.py`: compare old and corrected rule labels.
- `design_group_stratified_split.py`: create the final cycle-group split.
- `build_revision_grouped_dataset.py`: rebuild labels/prompts within cycle.
- `build_revision_boundary_comparison.py`: construct matched augmentation/weighting inputs.
- `train_revision_base_lora.py`: train Base-LoRA.
- `train_revision_comparison_lora.py`: train pure-augmentation and boundary-weighted LoRA.
- `evaluate_revision_lora.py`: candidate-token LoRA evaluation.
- `audit_revision_comparison.py`: audit the three LoRA variants.
- `build_revision_provenance.py`: hash private run artifacts.
- `revision_traditional_baselines.py`: Isolation Forest and Random Forest.
- `revision_audit_traditional_baselines.py`: recompute and audit traditional-baseline results.
- `revision_lstm_baseline_v3.py`: final LSTM implementation and run.
- `revision_evaluate_general_llm.py`: zero-/four-shot Qwen baselines.
- `revision_evaluate_rule_verifier_grouped.py`: RuleVerifier correction evaluation.
- `revision_benchmark_rule_verifier.py`: repeated CPU timing.
- `revision_evaluate_natural_near_threshold.py`: held-out-cycle near-threshold analysis.
- `revision_data_validity_analysis.py`: voltage redundancy, threshold coverage, and sensitivity.
- `revision_build_baseline_comparison_v2.py`: final eight-method comparison and statistics.
- `verify_paper_results.py`: public release-to-paper consistency check.

## Superseded or diagnostic-only

- `design_grouped_split.py`: earlier chronological split proposal; not the final split.
- `revision_build_baseline_comparison.py`: superseded by the `_v2` comparison.
- `smoke_test.py`: historical v3 adapter smoke test with old paths.

The root `scripts/20_rule_verifier_v3.py`, `scripts/21_build_boundary_testset.py`, and `scripts/23_build_boundary_train_v4.py` are historical components reused by the final revision.
