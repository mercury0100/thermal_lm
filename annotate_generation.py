#!/usr/bin/env python3
"""
Annotate generation with per-token aleatoric uncertainty (tau) for **Thermal LM**.

- Loads a checkpoint dir produced by train_thermal_lm.py (expects pytorch_model.bin + tokenizer/*).
- Generates token-by-token while extracting τ for each predicted token.
- Outputs either:
    * plain text with [tau=...] tags, or
    * color-annotated HTML (darker background = higher tau)

Examples:
python annotate_generation.py \
  --ckpt_dir checkpoints/thermal_best_ep2 \
  --prompt "The discovery suggests that" \
  --max_new_tokens 60 \
  --format html \
  --out_file annotated.html
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, set_seed

# Use the new module
from thermal_lm import ThermalLMForCausalLM


# ---------- Loading (Thermal LM) ----------

def load_thermal_model(ckpt_dir: str, device: torch.device):
    """
    Load tokenizer + ThermalLM skeleton, then load state_dict from the checkpoint.
    Checkpoints created by train_thermal_lm.py use torch.save(model.state_dict()).
    """
    ckpt = Path(ckpt_dir)
    weights = ckpt / "pytorch_model.bin"
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights file: {weights}")

    # Tokenizer was saved alongside the checkpoint
    tokenizer = AutoTokenizer.from_pretrained(ckpt.as_posix())

    # Build a fresh ThermalLM wrapper on the same base (gpt2 by default)
    model = ThermalLMForCausalLM.from_base_pretrained("gpt2")

    # Load weights
    state = torch.load(weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load warning] missing keys: {missing}")
    if unexpected:
        print(f"[load warning] unexpected keys: {unexpected}")

    model.to(device).eval()
    return model, tokenizer


# ---------- Generation + τ capture ----------

@torch.no_grad()
def generate_with_tau(
    model: ThermalLMForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    top_p: float = 0.95,
    temperature: float = 1.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Nucleus sampling (or greedy) with per-step τ annotation.
    Returns:
      full_text, list of (token_str, tau_value) for the generated tokens (not including the prompt).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    generated_pairs: List[Tuple[str, float]] = []

    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attention_mask)   # ModelOutput
        logits = out.logits                                              # (1,T,V)
        logtau = out.aux["logtau"]                                       # (1,T)
        tau = logtau.exp()

        last_logits = logits[:, -1, :] / max(1e-8, temperature)          # user sampling temp (independent of τ)
        probs = torch.softmax(last_logits, dim=-1)

        if top_p < 1.0:  # nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)       # (1,V)
            cum = torch.cumsum(sorted_probs, dim=-1)                             # (1,V)
            cutoff = torch.searchsorted(cum[0], torch.tensor(top_p, device=cum.device)).item()
            k = max(1, min(cutoff + 1, sorted_probs.size(-1)))
            keep_idx = sorted_idx[:, :k]                                         # (1,k)
            keep_probs = sorted_probs[:, :k] / sorted_probs[:, :k].sum(dim=-1, keepdim=True)
            next_id = keep_idx[0, torch.multinomial(keep_probs[0], num_samples=1)]
        else:
            next_id = torch.argmax(probs[0], dim=-1)

        last_tau = float(tau[0, -1].item())
        token_str = tokenizer.decode([next_id.item()], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        generated_pairs.append((token_str, last_tau))

        # Append and continue
        next_id = next_id.view(1, 1).to(device)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        attention_mask = torch.ones_like(input_ids, device=device)

    full_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return full_text, generated_pairs


# ---------- Rendering ----------

def render_text_annotated(pairs: List[Tuple[str, float]]) -> str:
    parts = []
    for tok, tau in pairs:
        clean = tok.replace("\n", "\\n")
        parts.append(f"{clean}[tau={tau:.3f}]")
    return "".join(parts)

def tau_to_hex_intensity(tau: float, tmin: float, tmax: float) -> str:
    """Map τ∈[tmin,tmax] to grayscale hex (lighter=low τ, darker=high τ)."""
    if tmax <= tmin:
        tmin, tmax = 0.8, 1.4
    x = max(tmin, min(tau, tmax))
    frac = (x - tmin) / (tmax - tmin + 1e-8)
    val = int(round(230 - 170 * frac))  # 230 -> 60
    return f"#{val:02x}{val:02x}{val:02x}"

def render_html_annotated(prompt: str, pairs: List[Tuple[str, float]], outfile: str):
    taus = [t for _, t in pairs]
    if len(taus) == 0:
        tmin, tmax = 0.9, 1.1
    else:
        ts = sorted(taus)
        lo = ts[max(0, int(0.10 * (len(ts) - 1)))]
        hi = ts[min(len(ts)-1, int(0.90 * (len(ts) - 1)))]
        tmin, tmax = lo, max(lo + 1e-6, hi)

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;"))

    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<style>body{font-family:ui-monospace,Menlo,Consolas,monospace;line-height:1.6;}",
        ".tok{padding:2px 3px;border-radius:4px;margin:1px;display:inline-block;}",
        ".legend{margin-top:12px;font-size:12px;color:#444}",
        ".prompt{color:#555;}</style>",
        "<h2>Aleatoric token shading (τ)</h2>",
        f"<div class='prompt'><b>Prompt:</b> {esc(prompt)}</div>",
        "<div style='margin-top:8px;'>"
    ]
    for tok, tau in pairs:
        color = tau_to_hex_intensity(tau, tmin, tmax)
        title = f"tau={tau:.4f}"
        html.append(f"<span class='tok' style='background:{color};' title='{title}'>{esc(tok)}</span>")
    html.append("</div>")
    html.append(f"<div class='legend'>Darker = higher τ (approx. aleatoric). "
                f"Scale from tokens: [{tmin:.3f}, {tmax:.3f}]</div>")

    Path(outfile).write_text("\n".join(html), encoding="utf-8")
    print(f"[HTML] wrote {outfile}")


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory with pytorch_model.bin and tokenizer")
    p.add_argument("--prompt", type=str, default="In summary,")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (separate from τ)")
    p.add_argument("--format", type=str, choices=["text", "html"], default="text")
    p.add_argument("--out_file", type=str, default="annotated.html")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_thermal_model(args.ckpt_dir, device)
    full_text, pairs = generate_with_tau(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        device=device
    )

    if args.format == "text":
        annotated = render_text_annotated(pairs)
        print("\n=== Prompt ===")
        print(args.prompt)
        print("\n=== Generated (token[tau]) ===")
        print(annotated)
        print("\n=== Full text ===")
        print(full_text)
    else:
        render_html_annotated(args.prompt, pairs, args.out_file)
        print("\nOpen in a browser:", args.out_file)


if __name__ == "__main__":
    main()