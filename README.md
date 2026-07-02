# Rule-Verified and Boundary-Aware LLM for Lithium-Ion Battery Anomaly Detection

This repository contains code and experimental artifacts for a rule-verified and boundary-aware LLM framework for lithium-ion battery anomaly detection.

## Main Idea

A LoRA-tuned LLM can achieve high accuracy on a standard random test set, but may still fail on numerical threshold-boundary and rule-applicability cases.

This project studies:

1. rule-supervised LoRA fine-tuning for battery anomaly detection;
2. executable RuleVerifier for rule consistency checking;
3. BoundarySet for threshold-boundary robustness evaluation;
4. boundary-aware LoRA training.

## Key Results

### Standard Test Set

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| v3 LoRA | 0.999188 | 0.998378 | 1.000000 | 0.999189 |
| v3 + RuleVerifier | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| v4 Boundary-Aware LoRA | 0.997023 | 0.994080 | 1.000000 | 0.997031 |

### BoundarySet

| Method | Accuracy | Precision | Recall | F1 | Wrong |
|---|---:|---:|---:|---:|---:|
| v3 LoRA | 0.605634 | 0.422222 | 0.904762 | 0.575758 | 28 |
| v3 + RuleVerifier | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 |
| v4 Boundary-Aware LoRA | 0.718310 | 0.515152 | 0.809524 | 0.629630 | 20 |
| v4 + RuleVerifier | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 |

## Important Files

- `scripts/20_rule_verifier_v3.py`: executable RuleVerifier.
- `scripts/21_build_boundary_testset.py`: build BoundarySet.
- `scripts/22_eval_v3_on_boundary.py`: evaluate v3 on BoundarySet.
- `scripts/23_build_boundary_train_v4.py`: build boundary-aware training samples.
- `scripts/24_train_qwen_lora_v4_boundary.py`: train v4 Boundary-Aware LoRA.
- `scripts/25_eval_v4_on_boundary.py`: evaluate v4 on BoundarySet.
- `scripts/26_eval_qwen_lora_v4_full_batched.py`: evaluate v4 on the standard test set.

## Data Notice

The raw battery dataset, pretrained base model, and trained LoRA checkpoints are not included in this repository.

The included BoundarySet and boundary-training samples are synthetic rule-based samples used for robustness evaluation and boundary-aware training.
