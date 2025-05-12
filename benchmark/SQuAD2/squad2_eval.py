import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def remove_punc(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    if not ground_truths:
        ground_truths = [""]

    return max(metric_fn(prediction, gt) for gt in ground_truths)


def load_squad(path: Path) -> List[Dict]:
    data = json.load(path.open(encoding='utf-8'))['data']
    examples: List[Dict] = []
    for article in data:
        for para in article['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                qid = qa['id']
                question = qa['question']
                answers = [ans['text'] for ans in qa.get('answers', [])]
                examples.append({'id': qid, 'context': context, 'question': question, 'answers': answers})
    return examples


def evaluate(ground_truth: Dict[str, List[str]], predictions: Dict[str, str]) -> Dict[str, float]:
    total = len(ground_truth)
    exact_sum = 0.0
    f1_sum = 0.0
    for qid, gts in ground_truth.items():
        pred = predictions.get(qid, "")
        exact = metric_max_over_ground_truths(exact_match_score, pred, gts)
        f1 = metric_max_over_ground_truths(f1_score, pred, gts)
        exact_sum += exact
        f1_sum += f1
    return {'exact_match': 100.0 * exact_sum / total, 'f1': 100.0 * f1_sum / total}


def evaluate_prediction_file(dev_file: Path, pred_file: Path) -> None:
    gt_list = load_squad(dev_file)
    gt = {ex['id']: ex['answers'] for ex in gt_list}
    preds = json.load(pred_file.open(encoding='utf-8'))

    metrics = evaluate(gt, preds)
    print(f"Exact Match: {metrics['exact_match']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}%")