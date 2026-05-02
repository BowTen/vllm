import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComparisonFailure:
    case_id: str
    reason: str
    reference: object | None = None
    actual: object | None = None


def load_jsonl_by_case_id(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "case_id" not in record:
                raise ValueError(
                    f"{path}:{line_number} is missing required case_id"
                )
            case_id = str(record["case_id"])
            if case_id in records:
                raise ValueError(f"{path}:{line_number} duplicate case_id {case_id}")
            records[case_id] = record
    return records


def compare_records(
    reference: dict[str, dict], actual: dict[str, dict]
) -> list[ComparisonFailure]:
    failures: list[ComparisonFailure] = []

    for case_id, reference_record in reference.items():
        actual_record = actual.get(case_id)
        if actual_record is None:
            failures.append(
                ComparisonFailure(
                    case_id=case_id,
                    reason="missing actual case",
                    reference=reference_record,
                    actual=None,
                )
            )
            continue

        if "output_token_ids" not in reference_record:
            failures.append(
                ComparisonFailure(
                    case_id=case_id,
                    reason="reference record missing output_token_ids",
                    reference=reference_record,
                    actual=actual_record,
                )
            )
            continue
        if "output_token_ids" not in actual_record:
            failures.append(
                ComparisonFailure(
                    case_id=case_id,
                    reason="actual record missing output_token_ids",
                    reference=reference_record,
                    actual=actual_record,
                )
            )
            continue

        reference_tokens = reference_record.get("output_token_ids")
        actual_tokens = actual_record.get("output_token_ids")
        if reference_tokens != actual_tokens:
            failures.append(
                ComparisonFailure(
                    case_id=case_id,
                    reason=_token_mismatch_reason(reference_tokens, actual_tokens),
                    reference=reference_tokens,
                    actual=actual_tokens,
                )
            )

    for case_id, actual_record in actual.items():
        if case_id not in reference:
            failures.append(
                ComparisonFailure(
                    case_id=case_id,
                    reason="unexpected actual case",
                    reference=None,
                    actual=actual_record,
                )
            )

    return failures


def compare_jsonl(reference_path: Path, actual_path: Path) -> list[ComparisonFailure]:
    return compare_records(
        load_jsonl_by_case_id(reference_path),
        load_jsonl_by_case_id(actual_path),
    )


def format_failures(failures: list[ComparisonFailure]) -> str:
    if not failures:
        return "DSSD outputs match exactly."
    lines = [f"{len(failures)} DSSD output comparison failure(s):"]
    for failure in failures:
        lines.append(f"- {failure.case_id}: {failure.reason}")
        if failure.reference is not None:
            lines.append(f"  reference: {failure.reference}")
        if failure.actual is not None:
            lines.append(f"  actual: {failure.actual}")
    return "\n".join(lines)


def _token_mismatch_reason(reference: Any, actual: Any) -> str:
    if isinstance(reference, list) and isinstance(actual, list):
        first_diff = _first_differing_index(reference, actual)
        return (
            "output_token_ids mismatch"
            f"; first differing index {first_diff}"
            f"; reference length {len(reference)}"
            f"; actual length {len(actual)}"
        )
    return "output_token_ids mismatch"


def _first_differing_index(reference: list, actual: list) -> int:
    for index, (reference_token, actual_token) in enumerate(zip(reference, actual)):
        if reference_token != actual_token:
            return index
    return min(len(reference), len(actual))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DSSD JSONL outputs by exact output_token_ids."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        failures = compare_jsonl(args.reference, args.actual)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 1
    output = format_failures(failures)
    if failures:
        print(output, file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
