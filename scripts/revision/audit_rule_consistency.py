from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import ijson


CHARGE_STATE = 110
DISCHARGE_STATE = 30


def number(item: dict, key: str) -> float | None:
    value = item.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def state_code(item: dict) -> int | None:
    value = number(item, "整车State状态（状态机编码）")
    return int(value) if value is not None else None


def evaluate(item: dict, previous: dict | None, restrict_common_to_active: bool) -> dict[str, bool]:
    state = state_code(item)
    total_voltage = number(item, "动力电池内部总电压V1")
    cell_voltage = number(item, "1号电池单体电压")
    current = number(item, "动力电池充/放电电流")
    temperature = number(item, "1号温度检测点温度")

    previous_total = number(previous, "动力电池内部总电压V1") if previous else None
    previous_cell = number(previous, "1号电池单体电压") if previous else None
    previous_current = number(previous, "动力电池充/放电电流") if previous else None
    previous_temperature = number(previous, "1号温度检测点温度") if previous else None

    total_diff = (
        abs(total_voltage - previous_total)
        if total_voltage is not None and previous_total is not None
        else None
    )
    cell_diff = (
        abs(cell_voltage - previous_cell)
        if cell_voltage is not None and previous_cell is not None
        else None
    )
    current_diff = (
        abs(current - previous_current)
        if current is not None and previous_current is not None
        else None
    )
    temperature_diff = (
        abs(temperature - previous_temperature)
        if temperature is not None and previous_temperature is not None
        else None
    )

    common_applicable = (
        state in {CHARGE_STATE, DISCHARGE_STATE}
        if restrict_common_to_active
        else True
    )
    return {
        "charge_total_jump": bool(
            state == CHARGE_STATE and total_diff is not None and total_diff > 3.0
        ),
        "charge_current_jump": bool(
            state == CHARGE_STATE and current_diff is not None and current_diff > 0.5
        ),
        "cell_voltage_jump": bool(
            common_applicable and cell_diff is not None and cell_diff > 0.05
        ),
        "temperature_jump": bool(
            common_applicable and temperature_diff is not None and temperature_diff > 3.0
        ),
        "discharge_total_limit": bool(
            state == DISCHARGE_STATE
            and total_voltage is not None
            and total_voltage > 378.2
        ),
    }


def main() -> None:
    project = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    source = (
        project
        / "data/processed/record_core_total_confirmed_for_new_error_anomalies.json"
    )

    counts = Counter()
    trigger_counts = Counter()
    unique_trigger_counts = Counter()
    mismatch_examples: list[dict] = []
    global_previous = None
    previous_by_cycle: dict[str, dict] = {}

    with source.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            counts["total"] += 1
            cycle = str(item.get("cycle_no", "missing"))
            stored = bool(item["is_anomaly"])
            global_rules = evaluate(item, global_previous, restrict_common_to_active=False)
            grouped_rules = evaluate(
                item, previous_by_cycle.get(cycle), restrict_common_to_active=False
            )
            prompt_rules = evaluate(
                item, previous_by_cycle.get(cycle), restrict_common_to_active=True
            )
            global_label = any(global_rules.values())
            grouped_label = any(grouped_rules.values())
            prompt_label = any(prompt_rules.values())

            counts["stored_anomaly"] += int(stored)
            counts["global_rule_anomaly"] += int(global_label)
            counts["grouped_rule_anomaly"] += int(grouped_label)
            counts["prompt_restricted_anomaly"] += int(prompt_label)
            counts["stored_vs_global_mismatch"] += int(stored != global_label)
            counts["stored_vs_grouped_mismatch"] += int(stored != grouped_label)
            counts["grouped_vs_prompt_restricted_mismatch"] += int(
                grouped_label != prompt_label
            )

            total_voltage = number(item, "动力电池内部总电压V1")
            cell_voltage = number(item, "1号电池单体电压")
            if total_voltage is not None and cell_voltage is not None:
                counts["both_voltage_fields_present"] += 1
                counts["voltage_fields_exactly_equal"] += int(
                    total_voltage == cell_voltage
                )

            for name, triggered in grouped_rules.items():
                trigger_counts[name] += int(triggered)
            active_triggers = [name for name, triggered in grouped_rules.items() if triggered]
            if len(active_triggers) == 1:
                unique_trigger_counts[active_triggers[0]] += 1

            if (
                len(mismatch_examples) < 20
                and (
                    stored != grouped_label
                    or grouped_label != prompt_label
                )
            ):
                mismatch_examples.append(
                    {
                        "record_id": item["record_id"],
                        "cycle_no": cycle,
                        "state": state_code(item),
                        "stored": stored,
                        "grouped_label": grouped_label,
                        "prompt_restricted_label": prompt_label,
                        "grouped_triggers": [
                            name for name, triggered in grouped_rules.items() if triggered
                        ],
                    }
                )

            global_previous = item
            previous_by_cycle[cycle] = item

    voltage_equal_rate = (
        counts["voltage_fields_exactly_equal"] / counts["both_voltage_fields_present"]
        if counts["both_voltage_fields_present"]
        else 0.0
    )
    report = {
        "source": str(source),
        "definitions": {
            "global_rule": "Original labeling code: previous row across the full ordered table.",
            "grouped_rule": "Same thresholds, but previous row is reset at each cycle boundary.",
            "prompt_restricted_rule": (
                "Grouped rule with cell-voltage and temperature jumps limited to "
                "charge/discharge states, matching the wording in the v3 prompt."
            ),
        },
        "counts": dict(counts),
        "trigger_counts_grouped": dict(trigger_counts),
        "unique_trigger_counts_grouped": dict(unique_trigger_counts),
        "voltage_field_exact_equality_rate": voltage_equal_rate,
        "mismatch_examples": mismatch_examples,
        "interpretation": {
            "charge_total_rule_unique_contribution": unique_trigger_counts[
                "charge_total_jump"
            ],
            "cell_voltage_rule_unique_contribution": unique_trigger_counts[
                "cell_voltage_jump"
            ],
            "note": (
                "If the observed total-voltage and cell-voltage fields are identical, "
                "a >3 V charge total-voltage jump is mathematically subsumed by the "
                ">0.05 V cell-voltage jump whenever both rules are applicable."
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(
        {
            "counts": report["counts"],
            "trigger_counts_grouped": report["trigger_counts_grouped"],
            "unique_trigger_counts_grouped": report["unique_trigger_counts_grouped"],
            "voltage_field_exact_equality_rate": voltage_equal_rate,
            "report_path": str(out_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
