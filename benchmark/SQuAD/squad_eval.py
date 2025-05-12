import json
import re
import string
import collections
from typing import Dict
import numpy as np
from scipy.optimize import linear_sum_assignment

def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def remove_punc(text):
        return text.translate(str.maketrans("", "", string.punctuation))
    def white_space_fix(text):
        return " ".join(text.split())
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(_normalize_answer(a_gold) == _normalize_answer(a_pred))

def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = _normalize_answer(a_gold).split()
    pred_toks = _normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)

def get_raw_scores(dataset, preds: Dict[str,str]):
    exact, f1 = {}, {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid = qa["id"]
                golds = [a["text"] for a in qa["answers"]]
                if not golds:
                    golds = [""]
                if qid not in preds:
                    continue
                pred = preds[qid]
                exact[qid] = max(compute_exact(g, pred) for g in golds)
                f1[qid]    = max(compute_f1(g, pred) for g in golds)
    return exact, f1

def apply_no_ans_threshold(scores, na_probs, has_ans_map, thresh: float):
    new = {}
    for qid, sc in scores.items():
        if na_probs.get(qid, 0.0) > thresh:
            new[qid] = float(not has_ans_map[qid])
        else:
            new[qid] = sc
    return new

def make_eval_dict(exact, f1, qids=None):
    if qids is None:
        total = len(exact)
        return collections.OrderedDict([
            ("exact", 100.0 * sum(exact.values()) / total),
            ("f1"   , 100.0 * sum(f1.values()) / total),
            ("total", total),
        ])
    else:
        total = len(qids)
        return collections.OrderedDict([
            ("exact", 100.0 * sum(exact[q] for q in qids) / total),
            ("f1"   , 100.0 * sum(f1[q]    for q in qids) / total),
            ("total", total),
        ])

def make_qid_map(dataset):
    m = {}
    for art in dataset:
        for p in art["paragraphs"]:
            for qa in p["qas"]:
                m[qa["id"]] = bool(qa["answers"])
    return m

def evaluate_prediction_file(
    gold_path: str,
    prediction_path: str,
    output_path: str = None,
    na_prob_thresh: float = 1.0
) -> Dict:

    with open(gold_path, encoding="utf-8") as f:
        data = json.load(f)["data"]
    with open(prediction_path, encoding="utf-8") as f:
        preds = json.load(f)

    has_ans = make_qid_map(data)
    na_probs = {qid: 0.0 for qid in preds}  # no-ans È®·ü ¾øÀ½

    exact_raw, f1_raw = get_raw_scores(data, preds)

    exact = apply_no_ans_threshold(exact_raw, na_probs, has_ans, na_prob_thresh)
    f1    = apply_no_ans_threshold(f1_raw   , na_probs, has_ans, na_prob_thresh)

    qid_list = list(has_ans.keys())
    overall = make_eval_dict(exact, f1)

    has_q = [q for q, v in has_ans.items() if v]
    no_q  = [q for q, v in has_ans.items() if not v]
    overall.update({f"HasAns_{k}": v for k, v in make_eval_dict(exact, f1, has_q).items()})
    overall.update({f"NoAns_{k}" : v for k, v in make_eval_dict(exact, f1, no_q).items()})

    print("¡æ SQuAD v2 1-shot Evaluation Results:")
    for k, v in overall.items():
        print(f"   {k:<10}: {v:6.2f}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(overall, f, indent=2)
    return overall
