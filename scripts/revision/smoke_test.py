from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    project = Path(sys.argv[1]).resolve()
    adapter_name = sys.argv[2]
    base_path = project / "models/Qwen/Qwen2___5-0___5B-Instruct"
    adapter_path = project / "outputs" / adapter_name
    test_path = project / "data/processed/lora_full_balanced_v3/test.jsonl"

    with test_path.open("r", encoding="utf-8") as handle:
        item = json.loads(next(line for line in handle if line.strip()))

    print(f"torch={torch.__version__}")
    print(f"cuda={torch.cuda.is_available()}")
    print(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"adapter={adapter_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    started = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path).cuda().eval()
    torch.cuda.synchronize()
    print(f"load_seconds={time.perf_counter() - started:.3f}")

    prompt = tokenizer.apply_chat_template(
        item["messages"][:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(
        model.device
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    print(f"inference_seconds={elapsed:.3f}")
    print(f"peak_memory_mib={torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    print(f"gold_is_anomaly={bool(item.get('is_anomaly'))}")
    print(f"generated={generated}")


if __name__ == "__main__":
    main()
