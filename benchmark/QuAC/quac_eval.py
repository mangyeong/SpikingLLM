import json
import re
from typing import Dict, Any, List, Tuple

def get_answer_text(entry: Any) -> str:
    if isinstance(entry, dict):
        return entry.get("text", "")
    return str(entry)

def load_quac(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[\.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def filter_by_question(candidates: List[str], question: str) -> List[str]:
    qs = set(re.findall(r"\w+", question.lower()))
    out = [s for s in candidates if qs & set(re.findall(r"\w+", s.lower()))]
    return out if out else candidates

def evaluate_quac(dataset: Dict[str, Any], pred_file: str) -> Dict[str, float]:
    with open(pred_file, "r", encoding="utf-8") as f:
        preds: List[Dict[str, Any]] = json.load(f)

    pred_map: Dict[Tuple[str,int], str] = {
        (p["id"], p["turn_id"]): p["answer"] for p in preds
    }

    total_em = 0.0
    total_f1 = 0.0
    count = 0

    def f1_score(pred: str, gold: str) -> float:
        p_tokens = pred.split()
        g_tokens = gold.split()
        common = set(p_tokens) & set(g_tokens)
        if not common:
            return 0.0
        prec = len(common) / len(p_tokens)
        rec  = len(common) / len(g_tokens)
        return 2 * prec * rec / (prec + rec)

    for dialog in dataset["data"]:
        for para in dialog["paragraphs"]:
            for qa in para["qas"]:
                key = (qa["id"], qa["turn_id"])
                raw_origs = qa.get("orig_answer", []) or qa.get("orig_answers", [])
                golds = [get_answer_text(a) for a in raw_origs]
                if not golds:
                    golds = [get_answer_text(a) for a in qa.get("answers", [])]

                pred = pred_map.get(key, "").strip()

                em = 1.0 if pred in golds else 0.0
                f1 = max((f1_score(pred, g) for g in golds), default=0.0)

                total_em += em
                total_f1 += f1
                count += 1

    return {
        "EM": 100.0 * total_em / count if count else 0.0,
        "F1": 100.0 * total_f1 / count if count else 0.0
    }
