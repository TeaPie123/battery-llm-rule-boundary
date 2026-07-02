# Reproducibility Guide

This document describes how to reproduce the main experiments in this repository.

## 1. Repository Scope

This repository provides the core code and experimental summaries for a rule-verified and boundary-aware LLM framework for lithium-ion battery anomaly detection.

The repository includes:

- rule-based anomaly verification code;
- BoundarySet construction code;
- boundary-aware training data construction code;
- v3 and v4 LoRA training/evaluation scripts;
- final experimental summaries.

The repository does not include:

- the original raw battery Excel dataset;
- the pretrained Qwen base model;
- trained LoRA checkpoints;
- large intermediate processed datasets.

Therefore, this repository is intended to support method reproduction and paper review. Full raw-data reproduction requires access to the original private battery dataset.

## 2. Expected Environment

The experiments were conducted with:

- Python 3.x
- PyTorch
- Transformers
- PEFT
- Datasets
- Pandas
- scikit-learn
- openpyxl
- accelerate

Minimal installation command:

    pip install torch transformers peft datasets pandas scikit-learn openpyxl accelerate safetensors

The base model used in the experiments was:

    Qwen2.5-0.5B-Instruct

In our local experiments, the model path was:

    models/Qwen/Qwen2___5-0___5B-Instruct

## 3. Data Preparation

The original raw battery dataset is not included in this repository.

The expected raw-data preprocessing pipeline is:

1. Convert the original battery record Excel file into a standardized processed CSV/XLSX format.
2. Apply the adapted rule-based labeling script.
3. Construct LoRA training, validation, and test datasets.

The adapted rule-labeling script is:

    python scripts/new_error_linux_full_total_confirmed.py

This script follows the senior rule-based anomaly detection logic, including:

- single-cell voltage jump threshold;
- current jump threshold;
- temperature jump threshold;
- charge total-voltage jump threshold;
- discharge total-voltage upper-limit rule.

## 4. v3 LoRA Baseline

The v3 model is the standard rule-supervised LoRA baseline trained on a full balanced dataset.

Build v3 full-balanced dataset:

    python scripts/13_prepare_lora_v3_full_balanced.py

Train v3 LoRA:

    python scripts/14_train_qwen_lora_v3_full_balanced.py

Check split overlap:

    python scripts/16_check_v3_split_overlap.py

Evaluate v3 on the standard test set:

    python scripts/18_eval_qwen_lora_v3_full_batched.py

Extract v3 wrong cases:

    python scripts/19_extract_v3_wrong_cases.py

## 5. RuleVerifier

The RuleVerifier is an executable rule-consistency checker.

It directly recomputes the anomaly label from structured numerical fields and compares the LLM prediction with the rule result.

Run:

    python scripts/20_rule_verifier_v3.py

The RuleVerifier is used to correct LLM mistakes caused by:

- wrong numerical threshold comparison;
- misunderstanding of rule applicability;
- confusing equality with strict greater-than conditions.

## 6. BoundarySet Evaluation

BoundarySet is a synthetic evaluation set designed to expose hidden LLM failures near rule thresholds and state applicability boundaries.

Build BoundarySet:

    python scripts/21_build_boundary_testset.py

The generated BoundarySet is saved at:

    data/processed/boundary_test/boundary_test.jsonl

Evaluate v3 on BoundarySet:

    python scripts/22_eval_v3_on_boundary.py

This evaluates:

- v3 LLM-only prediction;
- RuleVerifier-only prediction;
- LLM + RuleVerifier correction.

## 7. v4 Boundary-Aware LoRA

The v4 model adds boundary-aware synthetic samples into the training set.

Build boundary-aware training samples:

    python scripts/23_build_boundary_train_v4.py

The synthetic boundary-training data is saved at:

    data/processed/boundary_train/boundary_train.jsonl

Train v4 Boundary-Aware LoRA:

    python scripts/24_train_qwen_lora_v4_boundary.py

Evaluate v4 on BoundarySet:

    python scripts/25_eval_v4_on_boundary.py

Evaluate v4 on the standard test set:

    python scripts/26_eval_qwen_lora_v4_full_batched.py

## 8. Main Experimental Results

### 8.1 Standard Test Set

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| v3 LoRA | 0.999188 | 0.998378 | 1.000000 | 0.999189 |
| v3 + RuleVerifier | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| v4 Boundary-Aware LoRA | 0.997023 | 0.994080 | 1.000000 | 0.997031 |

### 8.2 BoundarySet

| Method | Accuracy | Precision | Recall | F1 | Wrong |
|---|---:|---:|---:|---:|---:|
| v3 LoRA | 0.605634 | 0.422222 | 0.904762 | 0.575758 | 28 |
| v3 + RuleVerifier | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 |
| v4 Boundary-Aware LoRA | 0.718310 | 0.515152 | 0.809524 | 0.629630 | 20 |
| v4 + RuleVerifier | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 |

## 9. Key Conclusion

The v3 LoRA model performs almost perfectly on the standard random test set, but its BoundarySet accuracy drops substantially.

This shows that ordinary random-test accuracy is insufficient for evaluating rule-based battery anomaly detection with LLMs.

BoundarySet reveals hidden failures in numerical threshold comparison and rule applicability.

Boundary-aware training improves boundary robustness, but does not fully eliminate LLM rule inconsistency.

The executable RuleVerifier provides a final rule-consistency guarantee.
