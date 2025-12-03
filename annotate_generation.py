#!/usr/bin/env python3
"""
Annotate generation with per-token aleatoric uncertainty (tau) for **Thermal LM**.

Supports two checkpoint styles:
  1) Thermal LM state_dict saved via your GPT-2 trainer (e.g., checkpoints_gpt2/thermal_best_ep*.pt)
  2) Full model directories (e.g., LLaMA) with pytorch_model.bin, tokenizer/*, train_args.json

If a checkpoint appears to be **baseline** (no Thermal head weights), we load a
standard AutoModelForCausalLM and expose tau=1.0 for all tokens so the tool
still works (you'll see uniform light shading).

Examples
--------
# LLaMA thermal checkpoint dir
python annotate_generation.py \
  --ckpt_dir checkpoints_llama/thermal_best_ep1 \
  --prompt "The discovery suggests that" \
  --max_new_tokens 60 --format html --out_file annotated.html

# GPT-2 thermal .pt checkpoint folder
python annotate_generation.py \
  --ckpt_dir checkpoints_gpt2 \
  --prompt "In summary, " --format html
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from thermal_lm import ThermalLMForCausalLM

# ----------------------- LoRA helper (optional) -----------------------

def maybe_attach_lora(model, ckpt_dir: Path, target_modules=("q_proj","k_proj","v_proj","o_proj")):
    """If train_args.json says use_lora=True, attach a matching PEFT LoRA before loading weights."""
    args_path = ckpt_dir / "train_args.json"
    if not args_path.exists():
        return model, False

    with args_path.open() as f:
        ta = json.load(f)

    if not ta.get("use_lora", False):
        return model, False

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "Checkpoint expects LoRA, but `peft` is not installed. `pip install peft`"
        ) from e

    r = int(ta.get("lora_rank", 8))
    alpha = int(ta.get("lora_alpha", 16))
    dropout = float(ta.get("lora_dropout", 0.1))

    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=list(target_modules),
    )
    model = get_peft_model(model, cfg)
    return model, True

# ----------------------- Loading + detection -----------------------

def detect_base_model(ckpt_dir: Path, default_base: str = "gpt2") -> str:
    """Try to find base model id from train_args.json; fallback to default."""
    args_path = ckpt_dir / "train_args.json"
    if args_path.exists():
        with args_path.open() as f:
            ta = json.load(f)
        return ta.get("base_model", ta.get("base_model_name_or_path", default_base))
    return default_base

def find_weights(ckpt_dir: Path) -> Path:
    """Support pytorch_model.bin (folder) or any *.pt under the folder."""
    bin_path = ckpt_dir / "pytorch_model.bin"
    if bin_path.exists():
        return bin_path
    pts = sorted(ckpt_dir.glob("*.pt"))
    if len(pts) == 0:
        raise FileNotFoundError(f"No weights found in {ckpt_dir} (missing pytorch_model.bin or *.pt).")
    pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0]

def is_thermal_state_dict(state: dict) -> bool:
    """Heuristic: Thermal checkpoints contain 'fc_logtau' parameters."""
    return any(k.startswith("fc_logtau.") for k in state.keys())

class _BaselineWithTau(torch.nn.Module):
    """Shim so baseline models expose `aux['tau']` as ones (no uncertainty field)."""
    def __init__(self, base_m: AutoModelForCausalLM):
        super().__init__()
        self.base = base_m
    def forward(self, *args, **kwargs):
        out = self.base(*args, **kwargs)
        logits = out.logits
        B, T, _ = logits.shape
        aux = {"tau": torch.ones(B, T, device=logits.device, dtype=logits.dtype)}
        setattr(out, "aux", aux)
        return out


def load_model_any(ckpt_dir: str, device: torch.device):
    ckpt = Path(ckpt_dir)
    weights = find_weights(ckpt)
    base_model = detect_base_model(ckpt, default_base="gpt2")

    # Tokenizer source: prefer the checkpoint folder if tokenizer artifacts exist
    tok_src = ckpt.as_posix() if any((ckpt / n).exists() for n in ("tokenizer.json","vocab.json","tokenizer.model")) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # Inspect state dict to decide whether it's Thermal or baseline
    state = torch.load(weights, map_location="cpu")
    as_thermal = is_thermal_state_dict(state)

    if (weights.name.endswith(".bin")):
        # Full HF model directory: prefer Thermal wrapper, but fallback to baseline shim if not Thermal
        model = ThermalLMForCausalLM.from_base_pretrained(base_model) if as_thermal else AutoModelForCausalLM.from_pretrained(base_model)
        # LoRA (if any) must be attached before loading
        model, used_lora = maybe_attach_lora(model, ckpt)
        missing, unexpected = [], []
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
        except Exception:
            # For full directories, rely on from_pretrained instead of raw state load
            model = AutoModelForCausalLM.from_pretrained(ckpt.as_posix()) if not as_thermal else model
        if not as_thermal and isinstance(model, AutoModelForCausalLM):
            model = _BaselineWithTau(model)
    else:
        # .pt state dicts from your training scripts
        if as_thermal:
            model = ThermalLMForCausalLM.from_base_pretrained(base_model)
            model, used_lora = maybe_attach_lora(model, ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
        else:
            base = AutoModelForCausalLM.from_pretrained(base_model)
            missing = base.load_state_dict(state, strict=False)[0]
            model = _BaselineWithTau(base)
            used_lora = False

    model.to(device).eval()
    print(f"[loaded] base={base_model} | thermal={as_thermal} | weights={weights.name}")
    return model, tokenizer

# ----------------------- Generation + τ capture (with KV cache) -----------------------

@torch.no_grad()
def generate_with_tau(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    top_p: float = 0.95,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Nucleus sampling (or greedy) with per-step τ annotation.
    Returns: (full_text, [(token_str, tau_value), ...]) for the generated tokens only.
    Optimized with KV cache to avoid O(T^2) recompute.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)

    pairs: List[Tuple[str, float]] = []
    past = None

    for step in range(max_new_tokens):
        out = model(input_ids=input_ids if past is None else input_ids[:, -1:],
                    attention_mask=attn if past is None else None,
                    past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :] / max(1e-8, temperature)

        # τ at current prediction slot
        tau_tensor = getattr(out, "aux", {}).get("tau", None)
        if tau_tensor is None:
            tau_last = 1.0
        else:
            tau_last = float(tau_tensor[0, -1].item())

        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            # first index where cum >= top_p
            k = int(torch.searchsorted(cum[0], torch.tensor(top_p, device=cum.device)).item()) + 1
            k = max(1, min(k, sorted_probs.size(-1)))
            keep_idx = sorted_idx[:, :k]
            keep_probs = sorted_probs[:, :k] / sorted_probs[:, :k].sum(dim=-1, keepdim=True)
            next_id = keep_idx[0, torch.multinomial(keep_probs[0], num_samples=1)]
        else:
            next_id = torch.argmax(probs[0], dim=-1)

        tok_str = tokenizer.decode([next_id.item()], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        pairs.append((tok_str, tau_last))

        # Extend context using KV cache
        next_id = next_id.view(1, 1).to(device)
        input_ids = next_id if past is not None else torch.cat([input_ids, next_id], dim=1)
        attn = None
        past = getattr(out, "past_key_values", None)

    # Reconstruct full text
    # If we used cache, the original prompt is not in input_ids anymore; re-decode explicitly
    full_text = prompt + tokenizer.decode([token_id for token, _ in pairs for token_id in tokenizer.encode(token, add_special_tokens=False)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return full_text, pairs

# ----------------------- Rendering -----------------------

def render_text_annotated(pairs: List[Tuple[str, float]]) -> str:
    safe = []
    for tok, tau in pairs:
        safe_tok = tok.replace("\n", "\\n")
        safe.append(f"{safe_tok}[tau={tau:.3f}]")
    return "".join(safe)


def tau_to_hex_intensity(tau: float, tmin: float, tmax: float) -> str:
    """Map τ∈[tmin,tmax] to grayscale hex (lighter=low τ, darker=high τ)."""
    if tmax <= tmin:
        tmin, tmax = 0.9, 1.1
    x = max(tmin, min(tau, tmax))
    frac = (x - tmin) / (tmax - tmin + 1e-8)
    val = int(round(230 - 170 * frac))  # 230 -> 60
    return f"#{val:02x}{val:02x}{val:02x}"


def render_html_annotated(prompt: str, pairs: List[Tuple[str, float]], outfile: str):
    taus = [t for _, t in pairs]
    if len(taus) == 0:
        tmin, tmax = 0.95, 1.10
    else:
        ts = sorted(taus)
        lo = ts[max(0, int(0.10 * (len(ts) - 1)))]
        hi = ts[min(len(ts)-1, int(0.90 * (len(ts) - 1)))]
        tmin, tmax = lo, max(lo + 1e-6, hi)

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<style>body{font-family:ui-monospace,Menlo,Consolas,monospace;line-height:1.6}"
        ".tok{padding:2px 3px;border-radius:4px;margin:1px;display:inline-block}"
        ".legend{margin-top:12px;font-size:12px;color:#444}.prompt{color:#555}</style>",
        "<h2>Aleatoric token shading (τ)</h2>",
        f"<div class='prompt'><b>Prompt:</b> {esc(prompt)}</div>",
        "<div style='margin-top:8px;'>"
    ]
    for tok, tau in pairs:
        color = tau_to_hex_intensity(tau, tmin, tmax)
        html.append(f"<span class='tok' style='background:{color}' title='tau={tau:.4f}'>" + esc(tok) + "</span>")
    html.append("</div>")
    html.append(f"<div class='legend'>Darker = higher τ (approx. aleatoric). Scale estimated from tokens: [{tmin:.3f}, {tmax:.3f}]</div>")
    Path(outfile).write_text("\n".join(html), encoding="utf-8")
    print(f"[HTML] wrote {outfile}")

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Checkpoint folder (with pytorch_model.bin or *.pt)")
    ap.add_argument("--prompt", default="The discovery suggests that")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--format", choices=["text","html"], default="html")
    ap.add_argument("--out_file", default="annotated.html")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_any(args.ckpt_dir, device)
    full_text, pairs = generate_with_tau(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        device=device,
    )

    if args.format == "text":
        print(render_text_annotated(pairs))
        print("\n=== Full text ===\n" + full_text)
    else:
        render_html_annotated(args.prompt, pairs, args.out_file)
        print("Open in a browser:", args.out_file)

if __name__ == "__main__":
    main()