# Reproducing the final revision experiments

## 1. Scope

The final paper uses cycle-isolated train/validation/test partitions and recomputes all five rule labels with the previous observation reset at every cycle boundary. The older random-record v3/v4 pipeline is retained only as historical evidence and must not be used to regenerate the final paper tables.

The full pipeline needs the private 614,497-record source file and the local Qwen2.5-0.5B-Instruct model. Public artifact checks do not need either.

## 2. Environment

The recorded run used Ubuntu 22.04, Python 3.12.11, one RTX 4090, PyTorch 2.13.0+cu132, Transformers 4.57.6, PEFT 0.19.1, Datasets 4.8.5, Accelerate 1.14.0, scikit-learn 1.9.0, pandas 2.3.3, SciPy 1.18.0, and NumPy 2.3.5.

Install the CUDA-compatible PyTorch build for the target machine, then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

The scripts expect the base model at:

```text
models/Qwen/Qwen2___5-0___5B-Instruct
```

## 3. Public result verification

Run this first after cloning:

```bash
python scripts/revision/verify_paper_results.py
```

It checks:

- Table II partition counts and prevalence;
- all displayed Table III metrics and exact BoundarySet confidence intervals;
- the three prespecified exact McNemar tests and Holm correction;
- Table IV natural near-threshold results;
- Table V synthetic case predictions and stored scores;
- Table VI RuleVerifier corrections and latency;
- voltage-field redundancy and rule-removal sensitivity.

## 4. Private input layout

Place the private processed record array at:

```text
data/processed/record_core_total_confirmed_for_new_error_anomalies.json
```

Each record must include `record_id`, `cycle_no`, `timestamp`, `step_status`, state code, total voltage, cell voltage, current, and temperature using the field names consumed by `build_revision_grouped_dataset.py`.

The stored old `is_anomaly` value is not trusted by the final pipeline; final labels are recomputed from the five rules within cycle.

## 5. Final pipeline order

Use an absolute repository path in place of `$PROJECT`.

### A. Audit the historical split and labels

```bash
python scripts/revision/audit_split_leakage.py "$PROJECT" "$PROJECT/outputs/revision_audit/split_leakage.json"
python scripts/revision/audit_rule_consistency.py "$PROJECT" "$PROJECT/outputs/revision_audit/rule_consistency.json"
```

These scripts document why the old random-record split and cross-cycle differencing were replaced.

### B. Create the final cycle-group split

```bash
python scripts/revision/design_group_stratified_split.py "$PROJECT" "$PROJECT/outputs/revision_audit/group_stratified_corrected_labels_design.json"
python scripts/revision/build_revision_grouped_dataset.py "$PROJECT" "$PROJECT/outputs/revision_audit/group_stratified_corrected_labels_design.json" "$PROJECT/data/processed/revision_grouped_v1"
```

The expected natural split counts are 430,170/92,133/92,194 records across 548/117/118 cycles. The builder refuses to overwrite an existing output directory.

### C. Build matched boundary interventions

The repository already contains the public synthetic BoundarySet and boundary-training scenarios. To construct the matched training inputs:

```bash
python scripts/revision/build_revision_boundary_comparison.py --project "$PROJECT" --seed 20260725 --boundary-weight 3
```

Pure augmentation appends 1,017 rows (three copies of each of 339 scenarios). Boundary weighted appends one copy of each scenario with weight 3, giving both interventions the same nominal boundary contribution.

### D. Train the three LoRA variants

```bash
python scripts/revision/train_revision_base_lora.py --project "$PROJECT" --train "$PROJECT/data/processed/revision_grouped_v1/train_balanced.jsonl" --val "$PROJECT/data/processed/revision_grouped_v1/val_balanced.jsonl" --output "$PROJECT/outputs/revision_grouped_v1/base_lora_seed42" --seed 42 --max-steps 1000 --learning-rate 1e-4 --max-length 1024

python scripts/revision/train_revision_comparison_lora.py --project "$PROJECT" --train "$PROJECT/data/processed/revision_boundary_comparison_v1/train_pure_augmentation.jsonl" --val "$PROJECT/data/processed/revision_grouped_v1/val_balanced.jsonl" --output "$PROJECT/outputs/revision_grouped_v1/pure_augmentation_seed42" --experiment pure_augmentation --seed 42 --max-steps 1000 --learning-rate 1e-4 --max-length 1024

python scripts/revision/train_revision_comparison_lora.py --project "$PROJECT" --train "$PROJECT/data/processed/revision_boundary_comparison_v1/train_boundary_weighted.jsonl" --val "$PROJECT/data/processed/revision_grouped_v1/val_balanced.jsonl" --output "$PROJECT/outputs/revision_grouped_v1/boundary_weighted_seed42" --experiment boundary_weighted --seed 42 --max-steps 1000 --learning-rate 1e-4 --max-length 1024
```

### E. Evaluate LoRA models

Run `evaluate_revision_lora.py` for each adapter against:

```text
data/processed/revision_grouped_v1/test_balanced.jsonl
data/processed/revision_grouped_v1/test_natural.jsonl
data/processed/boundary_test/boundary_test.jsonl
```

Use output names of the form `outputs/revision_grouped_v1/eval/<adapter>_<dataset>.jsonl`. Classification uses the next-token log-probability difference between the one-token Chinese labels “正常” and “异常” with threshold 0.5.

Then run:

```bash
python scripts/revision/audit_revision_comparison.py --project "$PROJECT"
python scripts/revision/build_revision_provenance.py --project "$PROJECT"
```

### F. Run comparison baselines

```bash
python scripts/revision/revision_traditional_baselines.py "$PROJECT"
python scripts/revision/revision_audit_traditional_baselines.py "$PROJECT"
python scripts/revision/revision_lstm_baseline_v3.py "$PROJECT"
python scripts/revision/revision_evaluate_general_llm.py --project "$PROJECT" --output-name revision_general_llm_baselines_v1 --batch-size 64 --max-length 1536
```

`revision_lstm_baseline_v3.py` is the final audited LSTM run despite the filename. General-Qwen evaluation must be invoked with `--max-length 1536`, matching the recorded run and the paper.

### G. RuleVerifier, natural-threshold, and validity analyses

```bash
python scripts/revision/revision_evaluate_rule_verifier_grouped.py "$PROJECT"
python scripts/revision/revision_benchmark_rule_verifier.py "$PROJECT"
python scripts/revision/revision_evaluate_natural_near_threshold.py "$PROJECT"
python scripts/revision/revision_data_validity_analysis.py "$PROJECT"
python scripts/revision/revision_build_baseline_comparison_v2.py "$PROJECT"
```

The RuleVerifier evaluation intentionally AST-loads the original functions from `scripts/20_rule_verifier_v3.py`. Its legacy regression guard additionally needs the old v3 test, prediction, and archived case files; those private record-level artifacts are not published.

## 6. Interpretation boundaries

- The 71-case BoundarySet is synthetic and fixed; its exact intervals describe this suite, not deployment prevalence.
- Natural near-threshold evidence covers only current and cell-voltage jumps.
- The total-voltage field duplicates the cell-voltage field in all 614,497 natural records, so total-voltage rules are validated only as synthetic logic probes.
- RuleVerifier reaches 100% agreement by executing the same label rules. It is a consistency layer, not an independent fault detector.
- The experiments use one source, one compact model, and one seed.
