# BoundarySet Evaluation Summary

## Purpose

BoundarySet is designed to evaluate whether a LoRA-tuned LLM can make numerically reliable decisions near rule thresholds in lithium-ion battery anomaly detection.

The tested boundary cases include:

1. charge current jump around 0.5A,
2. cell voltage jump around 0.05V,
3. temperature jump around 3℃,
4. charge total voltage jump around 3V,
5. discharge total voltage around 378.2V,
6. rule applicability under charge, discharge, and rest states.

## BoundarySet statistics

- Total samples: 71
- Normal samples: 50
- Anomaly samples: 21

## v3 LLM-only result on BoundarySet

- Accuracy: 0.6056338028169014
- Precision: 0.4222222222222222
- Recall: 0.9047619047619048
- F1: 0.5757575757575758
- TP: 19
- FP: 26
- TN: 24
- FN: 2

## RuleVerifier-only result on BoundarySet

- Accuracy: 1.0
- Precision: 1.0
- Recall: 1.0
- F1: 1.0
- TP: 21
- FP: 0
- TN: 50
- FN: 0

## LLM + RuleVerifier result on BoundarySet

- Accuracy: 1.0
- Precision: 1.0
- Recall: 1.0
- F1: 1.0
- TP: 21
- FP: 0
- TN: 50
- FN: 0

## Error count

- LLM wrong cases: 28
- LLM-rule inconsistent cases: 28
- Corrected by RuleVerifier: 28

## Accuracy by case type

| Case type | Total | Wrong | Accuracy |
|---|---:|---:|---:|
| cell_voltage_boundary | 24 | 3 | 0.875 |
| charge_current_boundary | 6 | 1 | 0.8333 |
| charge_total_voltage_jump_boundary | 5 | 3 | 0.4 |
| discharge_total_voltage_limit_boundary | 5 | 2 | 0.6 |
| non_charge_current_applicability | 8 | 8 | 0.0 |
| non_charge_total_voltage_jump_applicability | 4 | 2 | 0.5 |
| non_discharge_total_voltage_limit_applicability | 4 | 0 | 1.0 |
| temperature_boundary | 15 | 9 | 0.4 |

## Main finding

The v3 LoRA model performs well on the standard random test set but fails on many threshold-boundary and rule-applicability cases. This demonstrates that high standard-test accuracy does not guarantee numerical reliability or rule consistency in safety-critical battery anomaly detection.

## Paper-ready conclusion

The proposed BoundarySet exposes hidden numerical and rule-applicability weaknesses of the LoRA-tuned LLM. RuleVerifier corrects all detected boundary errors, showing that executable rule verification is necessary for reliable LLM-based battery anomaly detection.
