# main.py
import os
import tempfile
import time
from typing import Optional

from arc_eval_fast import eval_arc_challenge
from llama.generation import Llama

def main(
    # ckpt_dir: str = "/home/mangyeong/.llama/checkpoints/Llama3.2-1B/",
    # tokenizer_path: str = "/home/mangyeong/.llama/checkpoints/Llama3.2-1B/tokenizer.model",
    ckpt_dir: str = "/home/mangyeong/PycharmProjects/llama3/relu_finetuned_3B-1/",
    tokenizer_path: str = "/home/mangyeong/PycharmProjects/llama3/relu_finetuned_3B-1/tokenizer.model",
    data_dir: str = "/data/LLM_dataset/arc-challenge/",
    save_dir: str = "/home/mangyeong/test/results/arc-challenge",
    batch_size: int = 1,
    num_fewshot: int = 25,
    max_seq_len: int = 4096,
    max_gen_len: int = 100,
    model_parallel_size: Optional[int] = None,
    seed: int = 42,
    gpu: int = 4,
    mode: str = "classification"  ##generation
):
############################################################
    os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["RANK"]        = "0"
    os.environ["WORLD_SIZE"]  = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["TMPDIR"]      = "/home/mangyeong/.cache/data-gym-cache/"
    tempfile.tempdir          = "/home/mangyeong/.cache/data-gym-cache/"
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
    print("¢º Loaded LLaMA checkpoint. Starting ARC Challenge evaluation!")

    start_time = time.time()
    accuracy = eval_arc_challenge(
        llama,
        data_dir=data_dir,
        save_dir=save_dir,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        max_gen_len=max_gen_len,
        mode=mode,
    )
    total_time = time.time() - start_time
    print(f"¢º Finished in {total_time:.1f}s, accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
