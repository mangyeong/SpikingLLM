import pytest
import json

from drop_eval import _normalize_answer, answer_json_to_strings, get_metrics, evaluate_json

# ------------------ Tests for normalization ------------------
@pytest.mark.parametrize("input_str,expected", [
    ("The quick, brown fox!", "quick brown fox"),
    ("An apple a day", "apple day"),
    ("Hello   World", "hello world"),
    ("Don't-stop!", "dontstop"),
])
def test_normalize_answer(input_str, expected):
    assert _normalize_answer(input_str) == expected

# ------------------ Tests for answer_json_to_strings ------------------
@pytest.mark.parametrize("answer_json,expected_list", [
    ({"number": 42}, ["42"]),
    ({"spans": ["alpha", "beta"]}, ["alpha", "beta"]),
    ({"date": "2020-01-01"}, ["2020-01-01"]),
    ({"number": 7, "spans": ["x"]}, ["7", "x"]),
    ({"number": None, "spans": [], "date": None}, []),
])
def test_answer_json_to_strings(answer_json, expected_list):
    assert answer_json_to_strings(answer_json) == expected_list

# ------------------ Tests for get_metrics (EM/F1) ------------------
def test_get_metrics_exact_match():
    pred = ["Paris"]
    gold = ["Paris"]
    em, f1 = get_metrics(pred, gold)
    assert em == 1.0
    assert f1 == 1.0

def test_get_metrics_case_insensitive():
    pred = ["paris"]
    gold = ["Paris"]
    em, f1 = get_metrics(pred, gold)
    assert em == 1.0
    assert f1 == 1.0

def test_get_metrics_f1_partial():
    pred = ["new york city"]
    gold = ["York City"]
    em, f1 = get_metrics(pred, gold)
    assert em == 0.0
    # "york city" tokens = ["york","city"], pred tokens include both -> F1 = 1.0
    assert pytest.approx(f1, rel=1e-6) == 1.0

# ------------------ Tests for evaluate_json ------------------

def test_evaluate_json_simple(tmp_path):
    gold_data = [
        {"query_id": "q1", "answer": {"number": 1}},
        {"query_id": "q2", "answer": {"spans": ["yes"]}},
    ]
    gold_file = tmp_path / "gold.json"
    gold_file.write_text(json.dumps(gold_data))

    preds = {"q1": "1", "q2": "Yes"}
    pred_file = tmp_path / "pred.json"
    pred_file.write_text(json.dumps(preds))

    summary = evaluate_json(str(gold_file), preds)
    assert summary["overall"]["em"] == 1.0
    assert summary["overall"]["f1"] == 1.0
    for t in summary["by_type"]:
        assert summary["by_type"][t]["em"] == 1.0
        assert summary["by_type"][t]["f1"] == 1.0
