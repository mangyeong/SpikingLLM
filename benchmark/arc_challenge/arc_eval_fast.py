import os
import time
import csv
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
import numpy as np
from typing import List, Dict
from llama.generation import Llama

CHOICES = ["A", "B", "C", "D"]


def get_arc(data_dir: str):
    test_fp = os.path.join(data_dir, "data", "test-00000-of-00001.parquet")
    val_fp  = os.path.join(data_dir, "data", "validation-00000-of-00001.parquet")
    test_tbl = pq.read_table(test_fp)
    val_tbl  = pq.read_table(val_fp)

    questions_test = test_tbl.column("question").to_pylist()
    choices_test   = test_tbl.column("choices").to_pylist()
    answers_test   = test_tbl.column("answerKey").to_pylist()

    questions_val  = val_tbl.column("question").to_pylist()
    choices_val    = val_tbl.column("choices").to_pylist()
    answers_val    = val_tbl.column("answerKey").to_pylist()

    return (questions_test, choices_test, answers_test), \
           (questions_val, choices_val, answers_val)


def choose_best_by_loglikelihood(llama: Llama, question: str, fewshots: List[List[str]]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = llama.model.to(device).eval()
    tokenizer = llama.tokenizer

    prefix = ""
    for q_fs, a_fs, b_fs, c_fs, d_fs, ans_fs in fewshots:
        prefix += f"Question: {q_fs}\nA. {a_fs}\nB. {b_fs}\nC. {c_fs}\nD. {d_fs}\n"
        prefix += "The best answer is " + ans_fs + "\n\n"
    prefix += f"Question: {question}\nYour response should end with 'The best answer is [letter]'\n"
    prefix += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    prefix_ids = tokenizer.encode(prefix, bos=True, eos=False)
    best_score = -float('inf')
    best_choice = None

    for choice in CHOICES:
        cont = f" The best answer is {choice}"
        cont_ids = tokenizer.encode(cont, bos=False, eos=True)
        ids = torch.tensor([prefix_ids + cont_ids], device=device)
        with torch.no_grad():
            logits = model.forward(ids, 0)
            logprobs = F.log_softmax(logits, dim=-1)
        score = sum(
            logprobs[0, len(prefix_ids) + i - 1, tok].item()
            for i, tok in enumerate(cont_ids, start=1)
        )
        if score > best_score:
            best_score = score
            best_choice = choice

    return best_choice


@torch.no_grad()
def eval_arc_classification(
    llama: Llama,
    q_test: List[str],
    c_test: List[dict],
    a_test: List[str],
    q_val: List[str],
    c_val: List[dict],
    a_val: List[str],
    save_dir: str,
    num_fewshot: int = 0,
    batch_size: int = 8,
) -> float:

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = llama.model.to(device).eval()
    tokenizer = llama.tokenizer

    prefix_str = ""
    for i in range(num_fewshot):
        texts = c_val[i].get("text", [])
        if len(texts) < 4: texts += [""]*(4-len(texts))
        prefix_str += (
            f"Question: {q_val[i]}\n"
            f"A. {texts[0]}\nB. {texts[1]}\n"
            f"C. {texts[2]}\nD. {texts[3]}\n"
            f"The best answer is {a_val[i]}\n\n"
        )
    prefix_ids = tokenizer.encode(prefix_str, bos=True, eos=False)

    cont_strs = [f" The best answer is {ch}" for ch in CHOICES]
    cont_ids  = [tokenizer.encode(s, bos=False, eos=True) for s in cont_strs]

    pad_id = getattr(tokenizer, "eos_id", None)
    if pad_id is None:
        pad_id = tokenizer.encode("", eos=True)[-1]

    os.makedirs(save_dir, exist_ok=True)
    results = []
    correct_cnt = 0
    N = len(q_test)

    for start in tqdm(range(0, N, batch_size), desc="ARC(CL)>"):
        batch_idxs = list(range(start, min(start+batch_size, N)))
        seqs, info = [], []  # (q_idx, choice_idx, prefix_len)

        for j in batch_idxs:
            q_str = (
                f"Question: {q_test[j]}\n"
                "Your response should end with 'The best answer is [letter]'\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            q_ids = tokenizer.encode(q_str, bos=False, eos=False)
            base_ids = prefix_ids + q_ids
            for c_idx, cid in enumerate(cont_ids):
                seqs.append(base_ids + cid)
                info.append((j, c_idx, len(base_ids)))

        max_len = max(len(s) for s in seqs)
        batch_tokens = [s + [pad_id]*(max_len-len(s)) for s in seqs]
        inp = torch.tensor(batch_tokens, device=device)

        bsz, seqlen = inp.shape
        for m in model.modules():
            if hasattr(m, "cache_k") and hasattr(m, "cache_v"):
                old_k = m.cache_k  # shape = (old_bsz, old_seq, n_heads, head_dim)
                # new shape = (bsz, seqlen, n_heads, head_dim)
                m.cache_k = old_k.new_zeros((bsz, seqlen, *old_k.shape[2:]))
                m.cache_v = old_k.new_zeros((bsz, seqlen, *old_k.shape[2:]))

        logits = model.forward(inp, 0)      # [B, L, V]
        logps  = F.log_softmax(logits, dim=-1)

        scores: Dict[int, List[float]] = {j:[-1e9]*4 for j in batch_idxs}
        for idx, (q_idx, c_idx, pref_len) in enumerate(info):
            length = len(cont_ids[c_idx])
            s = 0.0
            for off in range(length):
                tok = seqs[idx][pref_len+off]
                s += logps[idx, pref_len+off-1, tok].item()
            scores[q_idx][c_idx] = s

        for j in batch_idxs:
            scs = scores[j]
            best = max(range(4), key=lambda x: scs[x])
            pred = CHOICES[best]
            correct = int(pred == a_test[j])
            correct_cnt += correct

            texts = c_test[j].get("text", [])
            if len(texts) < 4:
                texts += [""]*(4-len(texts))

            results.append([
                q_test[j],
                texts[0], texts[1], texts[2], texts[3],
                a_test[j],
                "",  # raw_output
                pred,
                correct,
                num_fewshot
            ])

    header = ["question","A","B","C","D","gold","raw_output","pred","correct","k_shot"]
    out_f = os.path.join(save_dir, "results.csv")
    with open(out_f, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(results)

    acc = correct_cnt / N
    print(f"¢º ARC(CL) Accuracy: {acc:.3f}")
    return acc

@torch.no_grad()
def eval_arc_batch(
    llama: Llama,
    q_test: List[str],
    c_test: List[dict],
    a_test: List[str],
    q_val: List[str],
    c_val: List[dict],
    a_val: List[str],
    save_dir: str,
    max_gen_len: int = 100,
    num_fewshot: int = 0,
    batch_size: int = 8,
) -> float:
    os.makedirs(save_dir, exist_ok=True)
    corrects, golds, raws, preds, ks = [], [], [], [], []
    start = time.time()
    N = len(q_test)

    def format_example(idx, include_answer):
        prompt = q_test[idx]
        for lab, txt in zip(c_test[idx]["label"], c_test[idx]["text"]):
            prompt += f"\n{lab}. {txt}"
        prompt += "\nAnswer [A/B/C/D]:"
        if include_answer:
            prompt += f" {a_test[idx]}\n\n"
        return prompt

    def gen_prompt(k):
        header = (
            "You are a multiple-choice question assistant.\n"
            "Answer with EXACTLY ONE uppercase letter: A, B, C, or D?and NOTHING ELSE.\n\n"
            "Here are some examples:\n\n"
        )
        for i in range(k):
            header += format_example(i, True)
        header += "Now answer the next question.\n"
        return header

    for i in tqdm(range(0, N, batch_size), desc="ARC>"):
        batch_prompts, batch_k, batch_labels, batch_cands = [], [], [], []
        for j in range(i, min(i + batch_size, N)):
            k = num_fewshot
            while k > 0:
                p = gen_prompt(k) + format_example(j, False)
                toks = llama.tokenizer.encode(p, bos=True, eos=False)
                if len(toks) + max_gen_len <= llama.model.params.max_seq_len:
                    break
                k -= 1
            if k < 0:
                raise ValueError(f"Index {j}: prompt too long even 0-shot.")
            batch_prompts.append(p)
            batch_k.append(k)
            batch_labels.append(a_test[j])
            batch_cands.append(c_test[j].get("text", []))

        outputs = llama.text_completion(
            prompts=batch_prompts,
            max_gen_len=max_gen_len,
            temperature=0,
            echo=False
        )

        for out, gold, k_val, cands in zip(outputs, batch_labels, batch_k, batch_cands):
            gen = out["generation"].strip()
            raws.append(gen); golds.append(gold); ks.append(k_val)

            if gen and gen[0].upper() in CHOICES:
                pred = gen[0].upper()
            else:
                mapped = None
                low = gen.lower()
                for idx, txt in enumerate(cands):
                    if txt.strip().lower() in low:
                        mapped = CHOICES[idx]
                        break
                pred = mapped or "UNKNOWN"
            preds.append(pred)
            corrects.append(pred == gold)

    acc = np.mean(corrects)
    elapsed = time.time() - start

    out_path = os.path.join(save_dir, "results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question","model_output","gold","predicted_choice","correct","K_shot"])
        for q, out, gold, pred, corr, k in zip(q_test, raws, golds, preds, corrects, ks):
            writer.writerow([q, out, gold, pred, corr, k])

    print(f"¢º ARC Challenge Accuracy: {acc:.3f}   (took {elapsed:.1f}s)")
    return acc


def eval_arc_challenge(
    llama: Llama,
    data_dir: str,
    save_dir: str,
    batch_size: int = 1,
    num_fewshot: int = 0,
    max_gen_len: int = 100,
    mode: str = "classification",
) -> float:
    (qt, ct, at), (qv, cv, av) = get_arc(data_dir)
    if mode == "classification":
        return eval_arc_classification(
            llama, qt, ct, at, qv, cv, av,
            save_dir, num_fewshot, batch_size
        )
    else:
        return eval_arc_batch(
            llama, qt, ct, at, qv, cv, av,
            save_dir, max_gen_len, num_fewshot, batch_size
        )
