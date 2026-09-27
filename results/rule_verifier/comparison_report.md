# RuleVerifier evaluation on cycle-isolated revision datasets

## Audit status

- The original `rule_verify` function was extracted by AST from `scripts/20_rule_verifier_v3.py`; no replacement rule implementation was introduced.
- Old-result regression: 3695 records matched the archived per-record outputs with zero mismatches; 3 archived corrections reproduced.
- RuleVerifier predictions matched the executable-rule gold labels for every balanced, natural, and BoundarySet record.
- Consequently, final accuracy is 100% by construction under the same rule specification. This is consistency enforcement, not independent evidence of real-world diagnostic accuracy.

## Natural-test corrections

| Method | Samples | LLM accuracy | LLM FP/FN | LLM–rule inconsistencies | Final accuracy |
|---|---:|---:|---:|---:|---:|
| Base-LoRA | 92194 | 99.9870% | 12/0 | 12 | 100.0000% |
| Pure augmentation | 92194 | 99.9989% | 1/0 | 1 | 100.0000% |
| Boundary weighted | 92194 | 99.9989% | 1/0 | 1 | 100.0000% |

## RuleVerifier latency

| Dataset | Samples | Elapsed (s) | Samples/s | µs/sample |
|---|---:|---:|---:|---:|
| balanced | 5310 | 0.115780 | 45862.71 | 21.80 |
| natural | 92194 | 2.067035 | 44602.04 | 22.42 |
| boundary | 71 | 0.001608 | 44165.35 | 22.64 |

## Interpretation boundary

RuleVerifier re-executes the same five-rule specification used to define the labels. Its value is deterministic consistency enforcement and correction accounting. It is not an independent learned baseline, it cannot detect anomalies omitted from the rules, and its 100% rule-label agreement must not be presented as external industrial validation.
