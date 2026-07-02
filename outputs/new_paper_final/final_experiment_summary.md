# Final Experiment Summary

## 1. Standard Test Set Results

Standard test set:
- Path: data/processed/lora_full_balanced_v3/test.jsonl / data/processed/lora_boundary_v4/test_clean.jsonl
- Total samples: 3695
- Normal samples: 1848
- Anomaly samples: 1847

| Method | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 LoRA | 0.999188 | 0.998378 | 1.000000 | 0.999189 | 1847 | 3 | 1845 | 0 |
| v3 + RuleVerifier | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1847 | 0 | 1848 | 0 |
| v4 Boundary-Aware LoRA | 0.997023 | 0.994080 | 1.000000 | 0.997031 | 1847 | 11 | 1837 | 0 |

## 2. BoundarySet Results

BoundarySet:
- Path: data/processed/boundary_test/boundary_test.jsonl
- Total samples: 71
- Normal samples: 50
- Anomaly samples: 21

| Method | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN | Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 LoRA | 0.605634 | 0.422222 | 0.904762 | 0.575758 | 19 | 26 | 24 | 2 | 28 |
| v3 + RuleVerifier | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 21 | 0 | 50 | 0 | 0 |
| v4 Boundary-Aware LoRA | 0.718310 | 0.515152 | 0.809524 | 0.629630 | 17 | 16 | 34 | 4 | 20 |
| v4 + RuleVerifier | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 21 | 0 | 50 | 0 | 0 |

## 3. BoundarySet Accuracy by Case Type

| Case Type | v3 Wrong / Total | v3 Accuracy | v4 Wrong / Total | v4 Accuracy |
|---|---:|---:|---:|---:|
| cell_voltage_boundary | 3 / 24 | 0.875 | 3 / 24 | 0.875 |
| charge_current_boundary | 1 / 6 | 0.8333 | 2 / 6 | 0.6667 |
| charge_total_voltage_jump_boundary | 3 / 5 | 0.4000 | 2 / 5 | 0.6000 |
| discharge_total_voltage_limit_boundary | 2 / 5 | 0.6000 | 2 / 5 | 0.6000 |
| non_charge_current_applicability | 8 / 8 | 0.0000 | 3 / 8 | 0.6250 |
| non_charge_total_voltage_jump_applicability | 2 / 4 | 0.5000 | 2 / 4 | 0.5000 |
| non_discharge_total_voltage_limit_applicability | 0 / 4 | 1.0000 | 0 / 4 | 1.0000 |
| temperature_boundary | 9 / 15 | 0.4000 | 6 / 15 | 0.6000 |

## 4. Main Findings

### Finding 1: Standard test accuracy is not enough

The v3 LoRA model achieves very high performance on the standard random test set:

- Accuracy: 99.9188%
- F1: 99.9189%

However, its accuracy drops to 60.5634% on BoundarySet.

This indicates that high standard-test accuracy does not guarantee numerical threshold robustness or rule consistency.

### Finding 2: BoundarySet exposes hidden model weaknesses

BoundarySet reveals that the LoRA model struggles with:

1. strict threshold comparison, such as 0.0500V vs > 0.05V;
2. equality-boundary cases, such as 0.5000A and 3.000℃;
3. rule applicability under different states, such as charge-only current-jump rules;
4. discharge voltage limit cases around 378.2V.

### Finding 3: RuleVerifier guarantees executable rule consistency

RuleVerifier corrects:

- 3 false positives on the standard test set;
- 28 errors of v3 on BoundarySet;
- 20 errors of v4 on BoundarySet.

Both v3 + RuleVerifier and v4 + RuleVerifier achieve 100% consistency with executable rule labels.

### Finding 4: Boundary-aware training partially improves robustness

v4 Boundary-Aware LoRA improves BoundarySet performance:

- Accuracy: 60.5634% -> 71.8310%
- F1: 57.5758% -> 62.9630%
- Wrong cases: 28 -> 20

The largest improvement appears in non-charge current applicability cases:

- v3: 0 / 8 correct
- v4: 5 / 8 correct

However, v4 slightly decreases standard-test precision:

- v3 FP: 3
- v4 FP: 11

Therefore, boundary-aware training improves model robustness but does not replace RuleVerifier.

## 5. Paper-Level Conclusion

The experiments show that a LoRA-tuned LLM can achieve near-perfect accuracy on a standard lithium-ion battery anomaly detection test set, but still fail on threshold-boundary and rule-applicability cases.

BoundarySet provides a targeted evaluation protocol for exposing such hidden weaknesses.

Boundary-aware training improves the model's own robustness, while RuleVerifier provides a final executable safety layer that guarantees rule-consistent decisions.

Thus, the proposed framework combines:

1. LoRA-based anomaly reasoning,
2. BoundarySet-based robustness evaluation,
3. boundary-aware training,
4. executable RuleVerifier correction.

