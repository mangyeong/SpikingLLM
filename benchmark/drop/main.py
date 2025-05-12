import os
import time
import json
from typing import Optional
from pathlib import Path
from tqdm import tqdm

from llama.generation import Llama
from drop_eval import evaluate_prediction_file, answer_json_to_strings


def main(
    ckpt_dir: str = "/home/mangyeong/.llama/checkpoints/Llama3.2-3B/",
    tokenizer_path: str = "/home/mangyeong/.llama/checkpoints/Llama3.2-3B/tokenizer.model",
    data_dir: str = "/data/LLM_dataset/DROP/",
    save_dir: str = "/home/mangyeong/test/results/relu_finetuned/",
    batch_size: int = 1,
    max_seq_len: int = 2048,
    max_gen_len: int = 10,
    num_fewshot: int = 3,  # 논문과 동일한 3-shot
    model_parallel_size: Optional[int] = None,
    seed: int = 1,
    gpu: int = 4,
):

############################################################
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
############################################################

    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        model_parallel_size=model_parallel_size,
        seed=seed,
        gpu=gpu,
    )
    print("✅ Loaded LLaMA checkpoint. Starting 3-shot DROP evaluation!")
    start_time = time.time()

    dev_path = Path(data_dir) / "dev.json"
    with open(dev_path, "r", encoding="utf-8") as f:
        raw = f.read()
        try:
            dev_data = json.loads(raw)
        except json.JSONDecodeError:
            dev_data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    test_data = dev_data
    predictions = {}
    for example in tqdm(test_data, desc="DROP Generation"):
        qid = example.get("query_id", example.get("question_id"))
        passage = example.get("paragraph", example.get("passage"))
        question = example["question"]

        k = num_fewshot
        while k >= 0:
            context = ""
            for shot in dev_data[:k]:
                shot_passage = shot.get("paragraph", shot.get("passage"))
                shot_question = shot["question"]
                shot_answers = answer_json_to_strings(shot.get("answer", {}))
                shot_answer = shot_answers[0] if shot_answers else ""
                context += (
                    f"Paragraph:\n{shot_passage}\n"
                    f"Question: {shot_question}\n"
                    f"Answer: {shot_answer}\n\n"
                )

            prompt = (
                context +
                f"Paragraph:\n{passage}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )

            token_ids = llama.tokenizer.encode(prompt, bos=True, eos=False)
            if len(token_ids) + max_gen_len <= max_seq_len:
                break
            k -= 1
        if k < 0:
            raise RuntimeError(f"Prompt too long even 0-shot for ID {qid}")

        output = llama.text_completion(
            prompts=[prompt],
            max_gen_len=max_gen_len,
            temperature=0,
            echo=False
        )
        predictions[qid] = output[0]["generation"].strip()

    os.makedirs(save_dir, exist_ok=True)
    pred_path = Path(save_dir) / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(predictions)} predictions in {time.time() - start_time:.1f}s.")

    metrics_path = Path(save_dir) / "metrics.json"
    evaluate_prediction_file(
        gold_path=str(dev_path),
        prediction_path=str(pred_path),
        output_path=str(metrics_path)
    )
    print(f"✅ DROP evaluation complete. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()