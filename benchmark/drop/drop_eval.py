import json
import re
import string
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------ Normalization ------------------
def _normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    def remove_punc(text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))
    def lower(text: str) -> str:
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ------------------ Parsing Gold Answers ------------------
def answer_json_to_strings(answer_json) -> List[str]:
    answers: List[str] = []
    if answer_json.get("number") is not None:
        answers.append(str(answer_json["number"]))
    if answer_json.get("spans"):
        for span in answer_json["spans"]:
            answers.append(span)
    if answer_json.get("date") is not None:
        answers.append(str(answer_json["date"]))
    return answers


# ------------------ Metrics Computation ------------------
def _compute_f1(a_pred: str, a_gold: str) -> float:
    pred_tokens = _normalize_answer(a_pred).split()
    gold_tokens = _normalize_answer(a_gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if set(gold_tokens).issubset(set(pred_tokens)):
        return 1.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def get_metrics(preds: List[str], golds: List[str]) -> Tuple[float, float]:
    norm_preds = [_normalize_answer(p) for p in preds]
    norm_golds = [_normalize_answer(g) for g in golds]
    em = float(any(p == g for p in norm_preds for g in norm_golds))
    n, m = len(golds), len(preds)
    if n == 0 and m == 0:
        return 1.0, 1.0
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = -_compute_f1(preds[j], golds[i])
    row_ind, col_ind = linear_sum_assignment(cost)
    total_f1 = -cost[row_ind, col_ind].sum() if len(row_ind)>0 else 0.0
    f1 = total_f1 / max(n, m)
    return em, f1


def evaluate_json(gold_path: str, pred_dict: Dict[str, str]) -> Dict:
    with open(gold_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    try:
        data = json.loads(''.join(lines))
    except json.JSONDecodeError:
        data = [json.loads(l) for l in lines]
    metrics = {"overall": {"em": [], "f1": []}, "by_type": {}}
    for ex in data:
        qid = ex.get("query_id", ex.get("question_id"))
        golds = answer_json_to_strings(ex["answer"])
        pred = pred_dict.get(qid, "")
        em, f1 = get_metrics([pred], golds)
        metrics["overall"]["em"].append(em)
        metrics["overall"]["f1"].append(f1)
        t = ex.get("answer_type", "span")
        metrics["by_type"].setdefault(t, {"em": [], "f1": []})
        metrics["by_type"][t]["em"].append(em)
        metrics["by_type"][t]["f1"].append(f1)
    summary = {"overall": {}, "by_type": {}}
    summary["overall"]["em"] = float(np.mean(metrics["overall"]["em"]))
    summary["overall"]["f1"] = float(np.mean(metrics["overall"]["f1"]))
    for t, vals in metrics["by_type"].items():
        summary["by_type"][t] = {"em": float(np.mean(vals["em"])), "f1": float(np.mean(vals["f1"]))}
    return summary


def evaluate_prediction_file(gold_path: str, prediction_path: str, output_path: str = None) -> Dict:
    with open(prediction_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    summary = evaluate_json(gold_path, preds)
    print("DROP Evaluation Results:")
    print(json.dumps(summary, indent=2))
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary
