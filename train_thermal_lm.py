#!/usr/bin/env python3
"""
Thermal LM Experiments (DistilGPT-2 only)

Variants:
    --variant baseline   Standard LM fine-tune (HF AutoModelForCausalLM)
    --variant thermal    Thermal LM: per-token hidden temperature (MC-free attenuation)

This script:
- trains/evals baseline vs. Thermal LM
- reports PPL, entropy, tau stats + corr(entropy, tau)
- computes calibration (ECE + reliability PNG)
- evaluates OOD perplexity on a chosen dataset
- estimates epistemic MI via dropout at eval-time (optional)
- saves checkpoints (best and/or periodic)

Requirements:
- thermal_lm.py (module with ThermalLMForCausalLM) in PYTHONPATH
"""

import os
import math
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, disable_progress_bar
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Import the Thermal LM module
from thermal_lm import ThermalLMForCausalLM

# Datasets progress bars off
disable_progress_bar()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BAR", "1")


# ---------------------------------------------------------------
#  Forward adapter (baseline / thermal)
# ---------------------------------------------------------------
def forward_three(model, tokens, attn, variant: str):
    """
    Standardize outputs across variants:
      returns (mu_like, logits, aux)
    - baseline: mu_like=logits_from_base, logits=logits_from_base, aux=zeros
    - thermal:  mu_like=logits,           logits=tempered_logits,  aux=logtau
    """
    out = model(input_ids=tokens, attention_mask=attn)  # HF (baseline) or ThermalLM ModelOutput

    if variant == "thermal":
        # ThermalLM returns a ModelOutput with attributes; τ is in out.aux["logtau"]
        logits = out.logits
        aux = getattr(out, "aux", None)
        if aux is not None and "logtau" in aux:
            logtau = aux["logtau"]
        else:
            # fallback zeros if aux missing (shouldn't happen)
            B, T, _ = logits.size()
            logtau = torch.zeros(B, T, device=logits.device, dtype=logits.dtype)
        return logits, logits, logtau

    # Baseline HF ModelOutput
    logits = out.logits
    B, T, V = logits.size()
    zeros = torch.zeros(B, T, device=logits.device, dtype=logits.dtype)
    return logits, logits, zeros


# ---------------------------------------------------------------
#  Evaluation (PPL, entropy, tau stats)
# ---------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device, variant):
    model.eval()
    total_nll, total_tokens = 0.0, 0

    # streaming Pearson correlation between token entropy and tau
    n = 0
    sx = sy = sx2 = sy2 = sxy = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        tokens = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        attn   = attention_mask[:, :-1].contiguous()

        mu_like, logits, aux = forward_three(model, tokens, attn, variant)
        B, T, V = logits.size()

        # NLL
        nll = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += B * T

        # entropy on predictive (tempered for thermal; same as logits)
        probs = torch.softmax(logits, dim=-1)
        token_ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)  # (B,T)

        # tau proxy
        token_proxy = aux.exp() if variant == "thermal" else torch.zeros(B, T, device=logits.device, dtype=logits.dtype)

        x = token_ent.reshape(-1).double()
        y = token_proxy.reshape(-1).double()
        nb = x.numel()
        n  += nb
        sx += x.sum().item()
        sy += y.sum().item()
        sx2 += (x * x).sum().item()
        sy2 += (y * y).sum().item()
        sxy += (x * y).sum().item()

    ppl = math.exp(total_nll / max(1, total_tokens))
    denom_x = n * sx2 - (sx ** 2)
    denom_y = n * sy2 - (sy ** 2)
    corr = (n * sxy - sx * sy) / math.sqrt(max(denom_x,1e-12) * max(denom_y,1e-12)) if (n > 1 and denom_x > 0 and denom_y > 0) else 0.0

    ent_mean = sx / max(1, n)
    proxy_mean = sy / max(1, n)
    return ppl, float(ent_mean), float(proxy_mean), float(corr)


# ---------------------------------------------------------------
#  Calibration: ECE & reliability diagram
# ---------------------------------------------------------------
@torch.no_grad()
def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """
    probs: (N, V) probabilities for each token
    labels: (N,) ground-truth token IDs
    """
    confidences, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    bins = []
    ece = 0.0
    edges = torch.linspace(0, 1, steps=n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i+1].item()
        mask = (confidences >= lo) & (confidences < hi if i < n_bins-1 else confidences <= hi)
        if mask.any():
            conf_bin = confidences[mask].mean().item()
            acc_bin  = correct[mask].mean().item()
            frac     = mask.float().mean().item()
            ece += abs(acc_bin - conf_bin) * frac
            bins.append({"bin": i, "conf": conf_bin, "acc": acc_bin, "frac": frac})
        else:
            bins.append({"bin": i, "conf": 0.0, "acc": 0.0, "frac": 0.0})
    return float(ece), bins


def reliability_diagram_png(bins, out_path: str):
    import matplotlib.pyplot as plt
    xs = [b["conf"] for b in bins if b["frac"] > 0]
    ys = [b["acc"]  for b in bins if b["frac"] > 0]
    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1])
    plt.scatter(xs, ys)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@torch.no_grad()
def eval_calibration(model, dataloader, device, variant, out_dir: str, tag: str = "val", max_batches: int = 200):
    model.eval()
    all_probs = []
    all_labels = []
    seen = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tokens = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        attn   = attention_mask[:, :-1]

        _, logits, _ = forward_three(model, tokens, attn, variant)
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.reshape(-1, probs.size(-1)))
        all_labels.append(labels.reshape(-1))
        seen += 1
        if seen >= max_batches:
            break

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    ece, bins = compute_ece(all_probs, all_labels, n_bins=15)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"reliability_{variant}_{tag}.png"
    reliability_diagram_png(bins, png_path.as_posix())
    print(f"[Calibration] ECE={ece:.4f} | diagram={png_path}")
    return ece, png_path.as_posix()


# ---------------------------------------------------------------
#  Data loaders (ID + OOD)
# ---------------------------------------------------------------
def build_dataloaders(tokenizer, block_size, batch_size):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=True)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {}
        for k in concatenated.keys():
            seq = concatenated[k][:total_len]
            result[k] = [seq[i : i + block_size] for i in range(0, total_len, block_size)]
        result["attention_mask"] = [[1] * block_size for _ in range(len(result["input_ids"]))]
        return result

    tokenized_train = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_val   = dataset["validation"].map(tokenize_fn, batched=True, remove_columns=["text"])

    train_ds = tokenized_train.map(group_texts, batched=True)
    val_ds   = tokenized_val.map(group_texts,   batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def build_ood_loader(tokenizer, block_size, batch_size, name="ag_news"):
    ds = load_dataset(name, split="test")

    def to_text(example):
        text = example.get("text") or example.get("content") or example.get("sentence") or ""
        if not text:
            text = " ".join(str(v) for v in example.values() if isinstance(v, str))
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=True)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    def group_texts(examples):
        keys = [k for k in examples.keys() if k in ("input_ids", "attention_mask")]
        concatenated = {k: sum(examples[k], []) for k in keys}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {}
        for k in keys:
            seq = concatenated[k][:total_len]
            result[k] = [seq[i:i+block_size] for i in range(0, total_len, block_size)]
        if "attention_mask" not in result:
            result["attention_mask"] = [[1]*block_size for _ in range(len(result["input_ids"]))]
        return result

    tokenized = tokenized.map(group_texts, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids","attention_mask"])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False, drop_last=False)


# ---------------------------------------------------------------
#  Epistemic MI with dropout (eval-time only)
# ---------------------------------------------------------------
@torch.no_grad()
def eval_mi_with_dropout(model, dataloader, device, variant, M: int = 20, max_batches: int = 64):
    model.train()  # enable dropout
    total_pred_ent = 0.0
    total_exp_ent  = 0.0
    total_tokens   = 0
    batches = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tokens = input_ids[:, :-1]; attn = attention_mask[:, :-1]

        probs_list = []
        for _ in range(M):
            _, logits, _ = forward_three(model, tokens, attn, variant)
            probs_list.append(torch.softmax(logits, dim=-1))
        probs_stack = torch.stack(probs_list, dim=0)  # (M,B,T,V)

        p_mean = probs_stack.mean(dim=0)
        pred_ent = -(p_mean * (p_mean.clamp_min(1e-12)).log()).sum(dim=-1)  # (B,T)
        cond_ent = -(probs_stack * (probs_stack.clamp_min(1e-12)).log()).sum(dim=-1).mean(dim=0)  # (B,T)

        total_pred_ent += pred_ent.sum().item()
        total_exp_ent  += cond_ent.sum().item()
        total_tokens   += pred_ent.numel()

        batches += 1
        if batches >= max_batches:
            break

    H = total_pred_ent / max(1, total_tokens)
    EH = total_exp_ent / max(1, total_tokens)
    MI = H - EH
    print(f"[Dropout-MI] predictive_entropy={H:.4f} | expected_entropy={EH:.4f} | MI={MI:.4f}")
    model.eval()
    return H, EH, MI


# ---------------------------------------------------------------
#  Checkpointing
# ---------------------------------------------------------------
def save_checkpoint(model, tokenizer, args, save_dir: str, epoch: int, tag: str, best_ppl: float = None):
    """
    Saves torch state_dict (works for both HF and ThermalLM models) + tokenizer + args.
    """
    save_path = Path(save_dir) / f"{args.variant}_{tag}_ep{epoch}"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "pytorch_model.bin")
    with open(save_path / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    tokenizer.save_pretrained(save_path.as_posix())
    with open(save_path / "README.txt", "w") as f:
        f.write(f"variant={args.variant} epoch={epoch} tag={tag} best_ppl={best_ppl}\n")
    print(f"[Checkpoint] Saved to {save_path}")


def load_model_from_checkpoint(base_model, ckpt_dir: str, device):
    state = torch.load(Path(ckpt_dir) / "pytorch_model.bin", map_location=device)
    base_model.load_state_dict(state, strict=False)
    return base_model


# ---------------------------------------------------------------
#  Train loop
# ---------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, variant, lambda_reg=1e-3):
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        tokens = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        attn   = attention_mask[:, :-1].contiguous()

        if variant == "thermal":
            out = model(input_ids=tokens, attention_mask=attn)  # ModelOutput
            logits = out.logits
            logtau = out.aux["logtau"]
            B, T, V = logits.size()
            ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="mean")
            reg = (logtau ** 2).mean()  # symmetric around logτ=0
            loss = ce + (lambda_reg * reg)
        else:
            out = model(input_ids=tokens, attention_mask=attn)  # HF ModelOutput
            logits = out.logits
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="mean")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


# ---------------------------------------------------------------
#  Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "thermal"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128, help="block size for chunking")
    parser.add_argument("--seed", type=int, default=42)

    # thermal-specific
    parser.add_argument("--lambda_reg", type=float, default=1e-3, help="log(tau) regularizer weight")

    # Saving / extra evals
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=0, help="Save every N epochs (0=only best)")
    parser.add_argument("--eval_calibration", action="store_true")
    parser.add_argument("--ood_dataset", type=str, default="", help="e.g., ag_news, imdb, tweet_eval")
    parser.add_argument("--eval_mi", action="store_true")
    parser.add_argument("--mi_samples", type=int, default=20)

    args = parser.parse_args()

    # Repro
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Running {args.variant.upper()} variant on distilgpt2 | device={device}")

    # Tokenizer + Data
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    train_loader, val_loader = build_dataloaders(tokenizer, args.max_length, args.batch_size)

    # Model
    if args.variant == "thermal":
        model = ThermalLMForCausalLM.from_base_pretrained("distilgpt2", lambda_reg=args.lambda_reg).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

    # Optimizer & scheduler
    if args.variant == "thermal":
        # slightly lower LR on the small head; keep weight_decay like baseline
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if not n.startswith("fc_logtau")], "lr": args.lr},
            {"params": [p for n, p in model.named_parameters() if n.startswith("fc_logtau")], "lr": args.lr * 0.5},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Train
    best_ppl = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.variant, lambda_reg=args.lambda_reg)
        ppl, ent, proxy, corr = evaluate(model, val_loader, device, args.variant)

        is_best = ppl < best_ppl
        if is_best:
            best_ppl = ppl
            save_checkpoint(model, tokenizer, args, args.save_dir, epoch, tag="best", best_ppl=best_ppl)
        if args.save_every and (epoch % args.save_every == 0):
            save_checkpoint(model, tokenizer, args, args.save_dir, epoch, tag="epoch", best_ppl=best_ppl)

        if args.variant == "thermal":
            print(f"[Epoch {epoch}] loss={loss:.4f} | ppl={ppl:.2f} | entropy={ent:.3f} | tau_mean={proxy:.5f} | corr(ent,tau)={corr:.3f}")
        else:
            print(f"[Epoch {epoch}] loss={loss:.4f} | ppl={ppl:.2f} | entropy={ent:.3f}")

    # Optional extra evals
    if args.eval_calibration:
        eval_calibration(model, val_loader, device, args.variant, out_dir=args.save_dir, tag="val")

    if args.ood_dataset:
        ood_loader = build_ood_loader(tokenizer, args.max_length, args.batch_size, name=args.ood_dataset)
        ppl_ood, ent_ood, proxy_ood, corr_ood = evaluate(model, ood_loader, device, args.variant)
        print(f"[OOD:{args.ood_dataset}] ppl={ppl_ood:.2f} | entropy={ent_ood:.3f} | "
              f"{'tau_mean' if args.variant=='thermal' else 'proxy_mean'}={proxy_ood:.5f} | corr(ent,proxy)={corr_ood:.3f}")
        if args.eval_calibration:
            eval_calibration(model, ood_loader, device, args.variant, out_dir=args.save_dir, tag=f"ood_{args.ood_dataset}")

    if args.eval_mi:
        eval_mi_with_dropout(model, val_loader, device, args.variant, M=args.mi_samples)
        if args.ood_dataset:
            ood_loader = build_ood_loader(tokenizer, args.max_length, args.batch_size, name=args.ood_dataset)
            eval_mi_with_dropout(model, ood_loader, device, args.variant, M=args.mi_samples)


if __name__ == "__main__":
    main()
