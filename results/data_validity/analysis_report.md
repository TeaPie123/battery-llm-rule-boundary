# Data-validity analysis for revision items 3-5

## Scope and terminology

- Natural observed data: 614,497 existing test records, partitioned by cycle into {'test': 92194, 'train': 430170, 'val': 92133}.
- Synthetic BoundarySet: 71 rule-generated stress-test cases.
- The natural observations are not external industrial boundary validation.

## Primary near-threshold coverage

The primary bands follow the BoundarySet perturbation scale. Counts across all observed records:

{
  "charge_total_jump": 0,
  "charge_current_jump": 272,
  "cell_voltage_jump": 185,
  "temperature_jump": 0,
  "discharge_total_limit": 0
}

See `natural_threshold_coverage.csv` for tight/primary/broad bands, sides of the threshold,
labels, states, and train/validation/test breakdowns.

## Voltage-field redundancy

- Current total-voltage/cell-voltage exact equality:
  614,497/614,497
  (100.000000%).
- Adjacent-difference exact equality:
  613,714/613,714
  (100.000000%).
- Charge-total-jump only triggers: 0.
- Charge-total and cell-jump simultaneous triggers: 0.
- Natural labels changed after removing both total-voltage rules:
  0.
- BoundarySet gold labels changed after removing both total-voltage rules:
  4/71.

## Interpretation guardrails

1. Near-threshold observed records support only in-source coverage analysis, not industrial deployment claims.
2. A zero count means the corresponding rule boundary is covered only synthetically in the present evidence.
3. Because total voltage and cell voltage are identical in the observed source mapping, independent PACK-level
   and cell-level physical conclusions are not identifiable.
4. Boundary model rescoring changes the gold definition only; it does not retrain or alter model predictions.
