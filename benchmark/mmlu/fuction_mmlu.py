import os
import time
import csv
from tqdm import tqdm
import torch
import torch.nn.functional as F

from llama.generation import Llama
import re

CHOICES = ["A", "B", "C", "D"]

def choose_best_by_loglikelihood(llama: Llama, question: str, choices: list[str], fewshots: list[list]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = llama.model.to(device).eval()
    tokenizer = llama.tokenizer

    prefix = ""
    for fs in fewshots:
        q_fs, a_fs, b_fs, c_fs, d_fs, ans_fs = fs
        prefix += f"Question: {q_fs}\nA. {a_fs}\nB. {b_fs}\nC. {c_fs}\nD. {d_fs}\n"
        prefix += "The best answer is " + ans_fs + "\n\n"
    prefix += f"Question: {question}\n"

    prefix_ids = tokenizer.encode(prefix, bos=True, eos=False)

    best_score = -float('inf')
    best_choice = None
    for i, choice in enumerate(choices):
        text_ids = prefix_ids + tokenizer.encode(f"The best answer is {choice}", bos=False, eos=True)
        ids = torch.tensor([text_ids], device=device)
        with torch.no_grad():
            logits = model.forward(ids, 0)  # [1, L, V]
            logprobs = F.log_softmax(logits, dim=-1)

        score = 0.0
        for idx, token_id in enumerate(text_ids[len(prefix_ids):], start=len(prefix_ids)):
            score += logprobs[0, idx-1, token_id].item()
        if score > best_score:
            best_score = score
            best_choice = choice
    return best_choice

def load_mc_csv(path: str):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError(f"[load_mc_csv] {path} line {i} has {len(row)} cols (expected 6)")
            rows.append(row)
    return rows

def format_subject(subject: str) -> str:
    return subject.replace("_", " ")

def format_example(example, include_answer=True):
    q,a,b,c,d,ans = example
    out = f"Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n"
    out += "Your response should end with 'The best answer is [letter]'\n"
    out += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    out += "The best answer is"
    if include_answer:
        out += f" {ans}\n\n"
    return out

def gen_prompt(dev, subject, k):
    header = f"Given the following multiple-choice questions about {format_subject(subject)}.\n\n"
    for i in range(k):
        header += format_example(dev[i], include_answer=True)
    return header

def map_answer_to_choice(output: str, choices_texts) -> str:
    out = output.strip().lower()
    for idx, txt in enumerate(choices_texts):
        if txt.strip().lower() in out:
            return CHOICES[idx]
    return "UNKNOWN"

# def map_answer_to_choice(output: str, choices_texts: list) -> str:
#     text = output.strip()
#     # 1) ¡°The best answer is C¡± ½ÄÀ¸·Î Ç¥±âµÈ °æ¿ì
#     m = re.search(r"answer\s+is\s*([ABCD])", text, re.IGNORECASE)
#     if m:
#         return m.group(1).upper()
#     # 2) ¸Ç ¾Õ ±ÛÀÚ°¡ A/B/C/D ÀÏ ¶§
#     if text and text[0].upper() in CHOICES:
#         return text[0].upper()
#     # 3) ¿ø·¡ ÀÖ´ø ÅØ½ºÆ® ¸ÅÄª (ÅØ½ºÆ® Æ÷ÇÔ ¿©ºÎ)
#     low = text.lower()
#     for i, choice_text in enumerate(choices_texts):
#         if choice_text.strip().lower() in low:
#             return CHOICES[i]
#     # ±× ¿Ü¿¡´Â UNKNOWN
#     return "UNKNOWN"

def eval_subject(
    llama: Llama,
    subject: str,
    dev: list,
    test: list,
    save_dir: str,
    num_fewshot: int,
    max_gen_len: int,
    batch_size: int,
    mode: str
):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    correct_cnt = 0

    for i in tqdm(range(0, len(test), batch_size), desc=f"Eval {subject}"):
        batch = test[i:i+batch_size]

        if mode == "classification":
            fewshots = dev[:num_fewshot]
            for example in batch:
                q, a, b, c, d, ans = example
                pred = choose_best_by_loglikelihood(
                    llama, q, CHOICES, fewshots
                )
                correct = (pred == ans)
                correct_cnt += int(correct)
                results.append([q, a, b, c, d, ans, "", pred, int(correct), num_fewshot])
        continue

        prompts, ks = [], []
        for example in batch:
            # k = len(dev)
            k = num_fewshot
            while k >= 0:
                pre = gen_prompt(dev, subject, k)
                post = format_example(example, include_answer=False)
                toks = llama.tokenizer.encode(pre+post, bos=True, eos=False)
                if len(toks) + max_gen_len <= llama.model.params.max_seq_len:
                    break
                k -= 1
            if k < 0:
                raise RuntimeError(f"Prompt too long even 0-shot for {subject}")
            ks.append(k)
            prompts.append(pre+post)

        outputs = llama.text_completion(
            prompts=prompts,
            max_gen_len=max_gen_len,
            temperature=0,
            echo=False
        )

        for example, k, out in zip(batch, ks, outputs):
            q,a,b,c,d,ans = example
            gen = out["generation"].strip()

            # if i == 0:  # Ã¹ ¹èÄ¡, Ã¹ »ùÇÃ¸¸ È®ÀÎ
            #     print(">>> DEBUG SAMPLE <<<")
            #     print("Question :", q)
            #     print("Choices  :", {"A": a, "B": b, "C": c, "D": d})
            #     print("Gold ans :", ans)
            #     # prompts ¸®½ºÆ®¿¡µµ ´ëÀÀµÇ´Â ¹®ÀÚ¿­À» ÀúÀåÇØµ×´Ù¸é °°ÀÌ Ãâ·Â
            #     prompt_str = prompts[0] if isinstance(prompts, list) else None
            #     print("Prompt   :", prompt_str)
            #     print("Raw gen  :", gen)
            #
            # if gen and gen[0].upper() in CHOICES:
            #     pred = gen[0].upper()
            # else:
            #     pred = map_answer_to_choice(gen, [a,b,c,d])

            # if pred == "UNKNOWN":
            #     print(f"[Warning][{subject} idx={i}] failed to map. raw generation:\n{gen}\n")

            correct = (pred == ans)
            if correct:
                correct_cnt += 1
            # results.append([q, a, b, c, d, ans, pred, int(correct), k])
            results.append([q, a, b, c, d, ans, gen, pred, int(correct), k])

    # header = ["question","A","B","C","D","gold","pred","correct","k_shot"]
    header = ["question", "A", "B", "C", "D", "gold", "raw_output", "pred", "correct", "k_shot"]

    with open(os.path.join(save_dir, f"{subject}.csv"), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    acc = correct_cnt / len(test)
    print(f"[{subject}] Accuracy: {acc:.4f}")
    return acc

def eval_all(
    data_dir: str,
    llama: Llama,
    save_dir: str,
    batch_size: int,
    num_fewshot: int,
    max_gen_len: int,
    mode: str
):
    subjects = sorted(
        fname.replace("_test.csv","")
        for fname in os.listdir(os.path.join(data_dir,"test"))
        if fname.endswith("_test.csv")
    )

    os.makedirs(save_dir, exist_ok=True)
    summary = []

    t0 = time.time()
    for subj in subjects:
        dev = load_mc_csv(os.path.join(data_dir,"dev", f"{subj}_dev.csv"))
        test= load_mc_csv(os.path.join(data_dir,"test",f"{subj}_test.csv"))
        acc = eval_subject(
            llama, subj, dev, test,
            save_dir=os.path.join(save_dir,subj),
            num_fewshot=num_fewshot,
            max_gen_len=max_gen_len,
            batch_size=batch_size,
            mode=mode
        )
        summary.append((subj, acc))

    mean_acc = sum(acc for _,acc in summary) / len(summary)
    with open(os.path.join(save_dir,"summary.csv"), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["subject","accuracy"])
        for subj, acc in summary:
            writer.writerow([subj, f"{acc:.4f}"])
        writer.writerow([])
        writer.writerow(["mean_accuracy", f"{mean_acc:.4f}"])

    print(f"\nAll done in {time.time()-t0:.1f}s, Mean Acc={mean_acc:.4f}")
    return mean_acc