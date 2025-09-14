import sys, os
print("sys.executable:", sys.executable)
print("sys.version:", sys.version)
import numpy as np
print("numpy:", np.__version__, np.__file__)
print("site-packages in sys.path:")
for p in sys.path:
    if "site-packages" in (p or ""):
        print("  ", p)

# save_gpt2xl_token_residuals.py
import os
import json
from tqdm import tqdm
import numpy as np
import torch
from transformer_lens import HookedTransformer

import sys, os
print("sys.executable:", sys.executable)
print("sys.version:", sys.version)
import numpy as np
print("numpy:", np.__version__, np.__file__)
print("site-packages in sys.path:")
for p in sys.path:
    if "site-packages" in (p or ""):
        print("  ", p)


def save_token_residuals(
    model_name="gpt2-xl",
    out_dir="gpt2xl_token_resids",
    device=None,
    batch_size=1024,
    store_fp16=True,
    activation_kind="hook_resid_post"   # change to "hook_resid_pre" or use accumulated_resid if you prefer
):
    os.makedirs(out_dir, exist_ok=True)
    # pick device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device} ...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # robustly get sizes
    vocab_size = len(model.tokenizer)  # safe for HF tokenizers
    # d_model: try a few fields
    d_model = getattr(model.cfg, "d_model", None) or getattr(model.cfg, "n_embd", None) or model.W_U.shape[0]
    n_layers = int(model.cfg.n_layers)

    print("Model dim is:", d_model)
    print("Number of layers is:", n_layers)

    # what layers to save? (one vector per transformer block output)
    layers = list(range(n_layers))

    dtype_np = np.float16 if store_fp16 else np.float32
    dtype_torch = torch.float16 if store_fp16 else torch.float32

    memmap_fname = os.path.join(out_dir, f"token_resids_{activation_kind}.memmap")
    # shape: (vocab, layers, d_model)
    shape = (vocab_size, len(layers), d_model)
    print(f"Creating memmap {memmap_fname}, shape {shape}, dtype {dtype_np} ...")
    mem = np.memmap(memmap_fname, dtype=dtype_np, mode="w+", shape=shape)

    # also save token strings (so you can look up token text)
    # use huggingface tokenizer helper:
    tokens_text = model.tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    meta = {
        "model_name": model_name,
        "vocab_size": vocab_size,
        "d_model": int(d_model),
        "n_layers": int(n_layers),
        "activation_kind": activation_kind,
        "memmap_filename": memmap_fname,
        "dtype": str(dtype_np),
        "token_strings_file": "token_strings.json"
    }

    # iterate in batches to avoid exploding memory
    with torch.no_grad():
        for start in tqdm(range(0, vocab_size, batch_size), desc="batches"):
            end = min(start + batch_size, vocab_size)
            batch_ids = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(1)  # shape (B, 1)
            # ensure we're passing integer token IDs (no automatic BOS prepending); pass exact tokens
            logits, cache = model.run_with_cache(batch_ids, return_cache_object=True, prepend_bos=True)

            # for each layer, extract the activation at position 0
            for li, L in enumerate(layers):
                act_name = f"blocks.{L}.{activation_kind}"
                if act_name not in cache:
                    raise KeyError(f"Activation {act_name} not present in cache; check activation_kind or model cfg.")
                # cache[act_name] has shape (batch, pos, d_model), pos==1 here
                arr = cache[act_name][:, 0, :].cpu().numpy()  # (B, d_model)
                mem[start:end, li, :] = arr.astype(dtype_np)

            # free cache references and clear CUDA memory if used
            del cache
            # if on cuda, free cached GPU memory (helps long runs)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # flush memmap and save metadata + token strings
    mem.flush()
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "token_strings.json"), "w", encoding="utf8") as f:
        json.dump(tokens_text, f, ensure_ascii=False)

    print("Done. memmap:", memmap_fname)
    print("Meta and token strings saved in", out_dir)


if __name__ == "__main__":
    save_token_residuals(
        model_name="gpt2-xl",
        out_dir="gpt2xl_token_resids",
        device="cuda",           # pick "cuda" or "cpu" if you want
        batch_size=1024,       # tune to your RAM/GPU memory
        store_fp16=True,
        activation_kind="hook_resid_post"
    )
