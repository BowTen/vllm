import json
from pathlib import Path

import pytest

from experiments.dssd_correctness import compare_outputs
from experiments.dssd_correctness.compare_outputs import (
    compare_jsonl,
    compare_records,
    format_failures,
    load_jsonl_by_case_id,
)
from experiments.dssd_correctness.run_dssd_cases import (
    _prompt_token_ids,
    _sampling_params,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_matching_jsonl_returns_no_failures(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.jsonl"
    actual_path = tmp_path / "actual.jsonl"
    records = [
        {"case_id": "case-a", "output_token_ids": [1, 2, 3]},
        {"case_id": "case-b", "output_token_ids": [4]},
    ]
    _write_jsonl(reference_path, records)
    _write_jsonl(actual_path, records)

    assert compare_jsonl(reference_path, actual_path) == []


def test_token_mismatch_reports_case_id_and_first_differing_index() -> None:
    failures = compare_records(
        {"case-a": {"case_id": "case-a", "output_token_ids": [1, 2, 3]}},
        {"case-a": {"case_id": "case-a", "output_token_ids": [1, 9, 3, 4]}},
    )

    assert len(failures) == 1
    formatted = format_failures(failures)
    assert "case-a" in formatted
    assert "first differing index 1" in formatted
    assert "reference length 3" in formatted
    assert "actual length 4" in formatted


def test_missing_actual_and_unexpected_actual_are_reported() -> None:
    failures = compare_records(
        {"missing": {"case_id": "missing", "output_token_ids": [1]}},
        {"extra": {"case_id": "extra", "output_token_ids": [1]}},
    )

    assert [(failure.case_id, failure.reason) for failure in failures] == [
        ("missing", "missing actual case"),
        ("extra", "unexpected actual case"),
    ]


def test_duplicate_case_id_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    _write_jsonl(
        path,
        [
            {"case_id": "dupe", "output_token_ids": [1]},
            {"case_id": "dupe", "output_token_ids": [2]},
        ],
    )

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_jsonl_by_case_id(path)


def test_missing_output_token_ids_is_reported() -> None:
    failures = compare_records(
        {"case-a": {"case_id": "case-a"}},
        {"case-a": {"case_id": "case-a"}},
    )

    assert len(failures) == 1
    assert failures[0].case_id == "case-a"
    assert failures[0].reason == "reference record missing output_token_ids"


def test_compare_main_handles_duplicate_case_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference_path = tmp_path / "reference.jsonl"
    actual_path = tmp_path / "actual.jsonl"
    _write_jsonl(
        reference_path,
        [
            {"case_id": "dupe", "output_token_ids": [1]},
            {"case_id": "dupe", "output_token_ids": [2]},
        ],
    )
    _write_jsonl(actual_path, [{"case_id": "dupe", "output_token_ids": [1]}])
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_outputs",
            "--reference",
            str(reference_path),
            "--actual",
            str(actual_path),
        ],
    )

    assert compare_outputs.main() == 1
    assert "duplicate case_id dupe" in capsys.readouterr().err


def test_run_dssd_cases_sampling_params_normalize_defaults() -> None:
    sampling_params = _sampling_params({"max_tokens": 8}, "case-a")

    assert sampling_params == {
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 0,
        "ignore_eos": True,
    }


def test_run_dssd_cases_sampling_params_require_max_tokens() -> None:
    with pytest.raises(ValueError, match="case case-a .* max_tokens"):
        _sampling_params({}, "case-a")


def test_run_dssd_cases_accepts_reference_output_sampling_shape() -> None:
    sampling_params = _sampling_params(
        {
            "case_id": "case-a",
            "prompt_token_ids": [1, 2],
            "seed": 99,
            "sampling": {
                "max_tokens": 8,
                "gamma": 4,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "ignore_eos": False,
            },
        },
        "case-a",
    )

    assert sampling_params == {
        "max_tokens": 8,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 99,
        "ignore_eos": False,
    }


def test_run_dssd_cases_rejects_prompt_only_case() -> None:
    with pytest.raises(ValueError, match="needs prompt_token_ids"):
        _prompt_token_ids({"prompt": "hello"}, "case-a")
