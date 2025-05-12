import os
import tempfile
import time
from typing import Optional
from llama.generation import Llama
from benchmark.mmlu.fuction_mmlu import eval_all
from model import ModelArgs

import torch
import torch.nn.functional as F


def main(
    # ckpt_dir: str = "/home/mangyeong/.llama/checkpoints/Llama3.1-8B/",
    # tokenizer_path: str = "/home/mangyeong/.llama/checkpoints/Llama3.1-8B/tokenizer.model",
    ckpt_dir: str = "/home/mangyeong/PycharmProjects/llama3/relu_finetuned_3B-1/",
    tokenizer_path: str = "/home/mangyeong/PycharmProjects/llama3/relu_finetuned_3B-1/tokenizer.model",
    data_dir: str = "/data/LLM_dataset/mmlu/data/",
    save_dir: str = "/home/mangyeong/test/results/relu_finetuned/relu/",
    batch_size: int = 1,
    num_fewshot: int = 5,
    max_seq_len: int = 3840,
    max_gen_len: int = 10,
    model_parallel_size: Optional[int] = None,
    seed: int = 42,
    gpu: int = 7,
    mode: str = "classification"
):
############################################################
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["TMPDIR"] = "/home/mangyeong/.cache/data-gym-cache/"
    tempfile.tempdir = "/home/mangyeong/.cache/data-gym-cache/"
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

    print("Using checkpoint dir:", ckpt_dir)
    print("Activation from params:", llama.model.params.activation_function)

    print("✅ Loaded LLaMA checkpoint. Starting MMLU evaluation!")
    t0 = time.time()

    accuracy = eval_all(
        data_dir=data_dir,
        llama=llama,
        save_dir=save_dir,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        max_gen_len=max_gen_len,
        mode=mode,
    )

    elapsed = time.time() - t0
    print(f"✅ MMLU evaluation complete.")
    print(f"✅ Mean Accuracy : {accuracy:.4f}")
    print(f"✅ Elapsed Time  : {elapsed:.1f}s")

if __name__ == "__main__":
    main()