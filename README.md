# Rule-Verified and Boundary-Aware Battery Anomaly Detection

Code and public experimental artifacts for the final revision of **“A Rule-Verified and Boundary-Aware Framework for Lithium-Ion Battery Anomaly Detection.”**

## Version notice

The final paper does **not** use the old random-record v3/v4 results as its main evidence. Those scripts remain in `scripts/` only for traceability. The final cycle-isolated experiment pipeline is in [`scripts/revision/`](scripts/revision/), and the corresponding aggregate results are in [`results/`](results/).

Three historical components are intentionally reused by the final pipeline:

- `scripts/20_rule_verifier_v3.py`: the original executable RuleVerifier implementation;
- `scripts/21_build_boundary_testset.py`: the 71-case synthetic BoundarySet builder;
- `scripts/23_build_boundary_train_v4.py`: the source of the 339 unique boundary-training scenarios.

The final LSTM filename also contains `v3`, but `revision_lstm_baseline_v3.py` is the audited final LSTM implementation; v1 and v2 were pilot runs and are not included.

## Final paper results

| Method | Balanced accuracy | Natural accuracy | Natural FP/FN | BoundarySet accuracy |
|---|---:|---:|---:|---:|
| Base-LoRA | 100.0000% | 99.9870% | 12/0 | 61.9718% |
| Pure augmentation | 100.0000% | 99.9989% | 1/0 | 80.2817% |
| Boundary weighted | 100.0000% | 99.9989% | 1/0 | 71.8310% |
| Isolation Forest | 98.1733% | 96.3620% | 3350/4 | 36.6197% |
| Random Forest | 99.9812% | 99.9881% | 11/0 | 74.6479% |
| LSTM | 99.5104% | 99.6345% | 322/15 | 56.3380% |
| Qwen zero-shot | 50.0000% | 2.8798% | 89539/0 | 29.5775% |
| Qwen four-shot | 50.0000% | 2.8798% | 89539/0 | 29.5775% |

The complete Table III metrics, exact confidence intervals, paired tests, natural near-threshold results, RuleVerifier corrections, timing, and data-validity aggregates are under [`results/`](results/).

## Repository map

- `scripts/revision/`: final cycle-isolated dataset construction, LoRA training/evaluation, conventional baselines, statistical audits, RuleVerifier evaluation, latency benchmark, and data-validity analysis.
- `scripts/13_*.py` through `scripts/26_*.py`: historical v3/v4 pipeline retained for provenance.
- `data/processed/boundary_test/`: public synthetic 71-case BoundarySet.
- `data/processed/boundary_train/`: public synthetic boundary-training scenarios.
- `results/paper_final/`: final comparison tables and synthetic per-case LoRA predictions.
- `results/near_threshold/`: aggregate held-out-cycle near-threshold analyses.
- `results/rule_verifier*`: correction and CPU-latency summaries.
- `results/data_validity/`: aggregate validity and sensitivity analyses.
- `docs/code_review.md`: code-review findings, fixes, and remaining limits.
- `docs/reproduce.md`: execution order and reproducibility instructions.

## Citation

If you download, use, or adapt any part of this repository—including the source code, synthetic datasets (`BoundarySet` and the boundary-training data), scripts, or released experimental artifacts—for academic research or publication, citation of our conference paper is required:

### Reference for Word (full-name format)

Copy the following reference directly into a Microsoft Word reference list:

> [1] ZiKang Wang, Weidong Wang, Chaohui Duan, and Yue Chen, “A Rule-Verified and Boundary-Aware Framework for Lithium-Ion Battery Anomaly Detection,” in *Proceedings of the 2026 China Automation Congress (CAC)*, Beijing, China, 2026. Accepted.

The paper has been accepted. Its page range and DOI have not yet been assigned; please use the final IEEE Xplore citation after the proceedings record becomes available.

## Quick public-artifact verification

```bash
python scripts/revision/verify_paper_results.py
```

A successful run prints `PAPER_RESULTS_CHECKS_OK`. This verifies the public artifacts against Tables II-VI and the paper’s data-validity statements without requiring private records or model checkpoints.

## Data and model availability

The raw battery records, record-level natural-test predictions, pretrained Qwen files, trained LoRA adapters, and model checkpoints are not published because of data-access, privacy, and size constraints. The repository includes only source code, synthetic data, synthetic per-case predictions, and aggregate natural-data results.

Expected local model path:

```text
models/Qwen/Qwen2___5-0___5B-Instruct
```

See [`docs/data_availability.md`](docs/data_availability.md) for the private inputs needed for a full rerun.

## Audit status

All 22 revision scripts parse successfully. The public result checker passes. The review found no error that changes the final headline metrics. Two non-invalidating reproducibility issues were documented and addressed in this release: automatic Holm correction was added to the final comparison builder, and cycle identifiers are normalized before duplicate detection. Two Table V scores in the paper are presentation-rounded rather than exact six-decimal renderings; the exact stored scores are published in `results/paper_final/representative_boundary_cases.csv`.

## License

No open-source license has been selected yet. Until the authors add one, normal copyright restrictions apply.
