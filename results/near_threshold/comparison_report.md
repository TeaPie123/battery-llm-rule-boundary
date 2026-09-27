# Natural near-threshold evaluation

## Selection audit

- Charge-current primary band: 43 held-out natural-test records at 0.5 ± 0.01 A.
- Cell-voltage primary band: 40 held-out natural-test records at 0.05 ± 0.001 V.
- Overlap between the two bands: 0 unique records.
- `all` includes records that may trigger other rules; `isolated` removes records with any non-target rule trigger.

## All near-threshold records

| Method | Current N | Current accuracy | Current FP/FN | Cell N | Cell accuracy | Cell FP/FN |
|---|---:|---:|---:|---:|---:|---:|
| Base-LoRA | 43 | 97.6744% | 1/0 | 40 | 100.0000% | 0/0 |
| Pure augmentation | 43 | 97.6744% | 1/0 | 40 | 100.0000% | 0/0 |
| Boundary weighted | 43 | 97.6744% | 1/0 | 40 | 100.0000% | 0/0 |
| Isolation Forest | 43 | 27.9070% | 31/0 | 40 | 50.0000% | 20/0 |
| Random Forest | 43 | 100.0000% | 0/0 | 40 | 72.5000% | 11/0 |
| LSTM | 43 | 34.8837% | 23/5 | 40 | 67.5000% | 12/1 |
| Qwen zero-shot | 43 | 27.9070% | 31/0 | 40 | 50.0000% | 20/0 |
| Qwen four-shot | 43 | 27.9070% | 31/0 | 40 | 50.0000% | 20/0 |

## Isolated target-rule records

| Method | Current N | Current accuracy | Current FP/FN | Cell N | Cell accuracy | Cell FP/FN |
|---|---:|---:|---:|---:|---:|---:|
| Base-LoRA | 43 | 97.6744% | 1/0 | 40 | 100.0000% | 0/0 |
| Pure augmentation | 43 | 97.6744% | 1/0 | 40 | 100.0000% | 0/0 |
| Boundary weighted | 43 | 97.6744% | 1/0 | 40 | 100.0000% | 0/0 |
| Isolation Forest | 43 | 27.9070% | 31/0 | 40 | 50.0000% | 20/0 |
| Random Forest | 43 | 100.0000% | 0/0 | 40 | 72.5000% | 11/0 |
| LSTM | 43 | 34.8837% | 23/5 | 40 | 67.5000% | 12/1 |
| Qwen zero-shot | 43 | 27.9070% | 31/0 | 40 | 50.0000% | 20/0 |
| Qwen four-shot | 43 | 27.9070% | 31/0 | 40 | 50.0000% | 20/0 |

## Interpretation guardrail

These subsets provide limited in-source evidence on naturally observed near-threshold records from held-out cycles. They do not establish cross-device, cross-domain, or industrial field generalization. Temperature and both total-voltage rule boundaries remain covered only by synthetic BoundarySet cases.
