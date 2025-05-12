import os
from datasets import load_dataset

def build_drop_splits(out_dir: str, seed: int = 42, test_ratio: float = 0.10):
    ds = load_dataset("drop")
    val = ds["validation"]
    splits = val.train_test_split(test_size=test_ratio, seed=seed)
    dev  = splits["train"]  # 90% dev.json
    test = splits["test"]   # 10% test.json

    os.makedirs(out_dir, exist_ok=True)
    dev.to_json(f"{out_dir}/dev.json")
    test.to_json(f"{out_dir}/test.json")
    print(f"Saved dev ({len(dev)}) ¡æ {out_dir}/dev.json")
    print(f"Saved test ({len(test)}) ¡æ {out_dir}/test.json")

if __name__ == "__main__":
    build_drop_splits(out_dir="/data/LLM_dataset/DROP/", seed=42, test_ratio=0.10)
