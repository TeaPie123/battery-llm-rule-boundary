# Traditional and temporal baseline comparison

## Audit status

All checks passed. All methods use the same cycle-isolated test record sets and gold labels.
The final LSTM v3 reached its best validation loss at epoch 56 and stopped naturally
at epoch 66 after patience 10; v1/v2 are convergence pilots
and must not be used in the paper.

## Headline results

| Method | Natural accuracy | Natural FP/FN | Natural AUPRC | Boundary accuracy | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Base-LoRA | 99.9870% | 12/0 | 99.9989% | 61.9718% | 44.8980% |
| Pure augmentation | 99.9989% | 1/0 | 100.0000% | 80.2817% | 70.8333% |
| Boundary weighted | 99.9989% | 1/0 | 100.0000% | 71.8310% | 52.3810% |
| Isolation Forest | 96.3620% | 3350/4 | 49.0966% | 36.6197% | 45.7831% |
| Random Forest | 99.9881% | 11/0 | 99.9187% | 74.6479% | 50.0000% |
| LSTM | 99.6345% | 322/15 | 99.1049% | 56.3380% | 20.5128% |
| Qwen zero-shot | 2.8798% | 89539/0 | 2.5734% | 29.5775% | 45.6522% |
| Qwen four-shot | 2.8798% | 89539/0 | 2.3864% | 29.5775% | 45.6522% |

## Interpretation

1. Random Forest is a strong in-source baseline because labels are deterministic threshold rules
   over the same numeric inputs. The paper cannot claim that LoRA universally outperforms
   traditional supervised models.
2. Pure augmentation and boundary weighting reduce natural-test false positives from Random
   Forest's 11 to 1, while keeping zero false negatives.
3. Pure augmentation has the highest BoundarySet point estimate, but Random Forest is competitive;
   all BoundarySet claims remain limited by 71 synthetic cases.
4. LSTM uses longer within-cycle context but performs worse than Random Forest and LoRA, especially
   on BoundarySet. Sequence context alone does not provide strict threshold compliance.
5. Isolation Forest has high recall but excessive natural false positives and poor BoundarySet
   specificity.
6. The unfinetuned general Qwen model predicts every sample as anomalous in both zero-shot and
   four-shot modes. This is a valid negative baseline showing that instruction prompting alone
   does not recover the deterministic battery rules; it must not be described as a full LLMAD
   reproduction.
