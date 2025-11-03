#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermal LM Experiments (LLaMA-1B on C4/C4-small only)

Tracks (fair comparisons):
  A) Frozen backbone
     - Baseline: train lm_head (per flag)
     - Thermal : same + τ-head

  B) LoRA-matched (identical LoRA on both)
     - Baseline-LoRA: LoRA params (+ lm_head as flagged)
     - Thermal-LoRA : same LoRA + τ-head

Features:
- base model via --base_model (e.g., meta-llama/Llama-3.2-1B)
- dataset via --train_dataset (c4, c4-small)
- optional freezing or LoRA (rank, alpha, dropout)
- thermal regularizer: (logτ**2).mean()
- grad accumulation, early stopping, ECE, OOD eval, dropout-MI

Requirements:
- transformers, datasets, torch, matplotlib
- peft (only if --use_lora)
- thermal_lm.py (Your ThermalLMForCausalLM module importable)
"""

import os
import math
import json
import random
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from datasets import load_dataset, disable_progress_bar
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from thermal_lm import ThermalLMForCausalLM

# Quiet datasets/tokenizers
disable_progress_bar()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BAR", "1")


# ===============================================================
# Helpers: LoRA, freezing, parameter groups
# ===============================================================

def try_import_peft():
    try:
        from peft import LoraConfig, get_peft_model
        return LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "PEFT is required when --use_lora is set. Install with `pip install peft`."
        ) from e


def add_lora(model, rank: int, alpha: int, dropout: float):
    """Attach LoRA to LLaMA-style attention projections."""
    LoraConfig, get_peft_model = try_import_peft()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    )
    return get_peft_model(model, cfg)


def freeze_backbone_for_fairness(model, variant: str, train_lm_head: str):
    """
    Freeze everything except:
      - (optionally) lm_head weights/bias depending on train_lm_head {none,bias,all}
      - (thermal only) τ-head (fc_logtau)
    """
    for p in model.parameters():
        p.requires_grad = False

    if variant == "thermal":
        for n, p in model.named_parameters():
            if n.startswith("fc_logtau"):
                p.requires_grad = True

    if train_lm_head in {"bias", "all"}:
        named = dict(model.named_parameters())
        for name in list(named.keys()):
            if "lm_head.bias" in name and train_lm_head in {"bias", "all"}:
                named[name].requires_grad = True
        if train_lm_head == "all":
            for name in list(named.keys()):
                if "lm_head.weight" in name:
                    named[name].requires_grad = True


def collect_param_groups(model, base_lr: float, wd: float,
                         variant: str, lora_lr: float = None, tau_lr_mult: float = 0.5):
    """Build optimizer param groups with distinct LRs for τ-head and LoRA."""
    groups = []
    lora_params, tau_params, base_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n or "lora_A" in n or "lora_B" in n:
            lora_params.append(p)
        elif variant == "thermal" and n.startswith("fc_logtau"):
            tau_params.append(p)
        else:
            base_params.append(p)

    if base_params:
        groups.append({"params": base_params, "lr": base_lr, "weight_decay": wd})
    if tau_params:
        groups.append({"params": tau_params, "lr": base_lr * tau_lr_mult, "weight_decay": wd})
    if lora_params:
        groups.append({"params": lora_params, "lr": lora_lr or base_lr, "weight_decay": wd})

    return groups


# ===============================================================
# Forward adapter (baseline / thermal)
# ===============================================================

def forward_three(model, tokens, attn, variant: str):
    """
    returns (logits, logits, aux)
    - baseline: logits, logits, zeros
    - thermal:  logits, logits, logtau
    """
    out = model(input_ids=tokens, attention_mask=attn)

    if variant == "thermal":
        logits = out.logits
        aux = getattr(out, "aux", None)
        if aux is not None and "logtau" in aux:
            logtau = aux["logtau"]
        else:
            B, T, _ = logits.size()
            logtau = torch.zeros(B, T, device=logits.device, dtype=logits.dtype)
        return logits, logits, logtau

    logits = out.logits
    B, T, _ = logits.size()
    zeros = torch.zeros(B, T, device=logits.device, dtype=logits.dtype)
    return logits, logits, zeros


# ===============================================================
# Evaluation (PPL, entropy, tau stats)
# ===============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, variant):
    model.eval()
    total_nll, total_tokens = 0.0, 0

    n = 0
    sx = sy = sx2 = sy2 = sxy = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        tokens = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        attn   = attention_mask[:, :-1].contiguous()

        _, logits, aux = forward_three(model, tokens, attn, variant)
        B, T, V = logits.size()

        nll = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += B * T

        probs = torch.softmax(logits, dim=-1)
        token_ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)  # (B,T)
        token_proxy = aux.exp() if variant == "thermal" else torch.zeros_like(token_ent)

        x = token_ent.reshape(-1).double()
        y = token_proxy.reshape(-1).double()
        nb = x.numel()
        n  += nb
        sx += x.sum().item(); sy += y.sum().item()
        sx2 += (x * x).sum().item(); sy2 += (y * y).sum().item()
        sxy += (x * y).sum().item()

    ppl = math.exp(total_nll / max(1, total_tokens))
    denom_x = n * sx2 - (sx ** 2)
    denom_y = n * sy2 - (sy ** 2)
    corr = (n * sxy - sx * sy) / math.sqrt(max(denom_x,1e-12) * max(denom_y,1e-12)) if (n > 1 and denom_x > 0 and denom_y > 0) else 0.0
    ent_mean = sx / max(1, n)
    proxy_mean = sy / max(1, n)
    return ppl, float(ent_mean), float(proxy_mean), float(corr)


# ===============================================================
# Calibration (ECE & reliability diagram)
# ===============================================================

@torch.no_grad()
def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
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
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.title("Reliability Diagram"); plt.tight_layout()
    plt.savefig(out_path); plt.close()


@torch.no_grad()
def eval_calibration(model, dataloader, device, variant, out_dir: str, tag: str = "val", max_batches: int = 200):
    model.eval()
    all_probs, all_labels = [], []
    seen = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tokens = input_ids[:, :-1]; labels = input_ids[:, 1:]; attn = attention_mask[:, :-1]
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


# ===============================================================
# Data loaders (C4 / C4-small ONLY)
# ===============================================================

def build_train_val(tokenizer, name: str, block_size: int, batch_size: int):
    """
    C4 / C4-small only.
    Streams text, concatenates tokens, and emits fixed-length blocks (no PAD inflation).
    Returns DataLoaders whose batches are dicts with torch tensors: input_ids, attention_mask.
    """
    import torch
    from itertools import islice
    from torch.utils.data import DataLoader, TensorDataset
    from datasets import load_dataset

    name = name.lower()
    assert name in ["c4", "c4-small"], f"Unsupported dataset: {name}"

    # choose number of blocks (not docs!) to materialize
    if name == "c4":
        train_blocks, val_blocks = 50000, 2000
    else:  # c4-small
        train_blocks, val_blocks = 10000, 1000

    print(f"[Data] Loading {name} (streaming) | blocks: train={train_blocks}, val={val_blocks} | block_size={block_size}")
    ds = load_dataset("allenai/c4", "realnewslike", streaming=True)

    # small helper: stream → pack into fixed-size token blocks
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def stream_blocks(example_iter, n_blocks):
        """
        Concatenate tokenized docs with EOS separators, and yield exactly n_blocks tensors
        of length block_size each (no padding).
        """
        buf = []
        yielded = 0
        for ex in example_iter:
            text = ex.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=True)
            if eos_id:
                ids.append(eos_id)  # doc separator
            buf.extend(ids)
            while len(buf) >= block_size and yielded < n_blocks:
                block = buf[:block_size]
                buf = buf[block_size:]
                yield torch.tensor(block, dtype=torch.long)
                yielded += 1
                if yielded >= n_blocks:
                    break

    # shuffled independent streams for train/val
    train_iter = ds["train"].shuffle(seed=42)
    val_iter   = ds["train"].shuffle(seed=43)

    # materialize the requested number of blocks
    train_blocks_t = list(islice(stream_blocks(train_iter, train_blocks), train_blocks))
    val_blocks_t   = list(islice(stream_blocks(val_iter,   val_blocks),   val_blocks))

    if len(train_blocks_t) < train_blocks or len(val_blocks_t) < val_blocks:
        print(f"[Data][warn] Fewer blocks than requested: train={len(train_blocks_t)}/{train_blocks}, "
              f"val={len(val_blocks_t)}/{val_blocks}")

    train_input_ids = torch.stack(train_blocks_t)  # (N, L)
    val_input_ids   = torch.stack(val_blocks_t)    # (M, L)
    train_attn      = torch.ones_like(train_input_ids)
    val_attn        = torch.ones_like(val_input_ids)

    train_ds = TensorDataset(train_input_ids, train_attn)
    val_ds   = TensorDataset(val_input_ids,   val_attn)

    def collate(batch):
        input_ids = torch.stack([b[0] for b in batch])
        attn      = torch.stack([b[1] for b in batch])
        return {"input_ids": input_ids, "attention_mask": attn}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)

    print(f"[Data] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader

def build_ood_loader(tokenizer, block_size, batch_size, name="ag_news"):
    """Optional OOD loader (kept for eval completeness)."""
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


# ===============================================================
# Dropout MI (epistemic probe)
# ===============================================================

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


# ===============================================================
# Train epoch (with grad accumulation)
# ===============================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, variant,
                lambda_reg=1e-3, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        tokens = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        attn   = attention_mask[:, :-1].contiguous()

        if variant == "thermal":
            out = model(input_ids=tokens, attention_mask=attn)
            logits = out.logits
            logtau = out.aux["logtau"]
            B, T, V = logits.size()
            ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="mean")
            reg = (logtau ** 2).mean()
            loss = ce + (lambda_reg * reg)
        else:
            out = model(input_ids=tokens, attention_mask=attn)
            logits = out.logits
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="mean")

        loss = loss / grad_accum_steps
        loss.backward()

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps
        steps += 1

    return total_loss / max(1, steps)


# ===============================================================
# Main
# ===============================================================

def main():
    p = argparse.ArgumentParser()
    # Experiment axes
    p.add_argument("--variant", type=str, default="baseline", choices=["baseline", "thermal"])
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B",
                   help="HF model id, e.g., 'meta-llama/Llama-3.2-1B'")
    p.add_argument("--train_dataset", type=str, default="c4-small",
                   choices=["c4", "c4-small"])

    # Train hyperparams
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--grad_accum_steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stop_patience", type=int, default=0, help="0=off")

    # Thermal
    p.add_argument("--lambda_reg", type=float, default=1e-3)
    p.add_argument("--tau_clamp_min", type=float, default=-2.0)
    p.add_argument("--tau_clamp_max", type=float, default=1.5)

    # Fairness knobs
    p.add_argument("--freeze_backbone", action="store_true",
                   help="freeze all except lm_head (per flag) and τ-head")
    p.add_argument("--train_lm_head", type=str, default="all",
                   choices=["none", "bias", "all"],
                   help="when backbone is frozen or LoRA: which parts of lm_head to train on BOTH variants")

    # LoRA (optional)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_lr", type=float, default=1e-4)

    # Saving / eval
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--eval_calibration", action="store_true")
    p.add_argument("--ood_dataset", type=str, default="", help="e.g., ag_news")
    p.add_argument("--eval_mi", action="store_true")
    p.add_argument("--mi_samples", type=int, default=20)

    args = p.parse_args()

    # Repro
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Running {args.variant.upper()} | base={args.base_model} | data={args.train_dataset} | device={device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Data
    train_loader, val_loader = build_train_val(tokenizer, args.train_dataset, args.max_length, args.batch_size)

    # Model
    if args.variant == "thermal":
        model = ThermalLMForCausalLM.from_base_pretrained(
            args.base_model,
            tau_clamp_min=args.tau_clamp_min,
            tau_clamp_max=args.tau_clamp_max,
            lambda_reg=args.lambda_reg,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)

    # Freeze backbone (Track A)
    if args.freeze_backbone:
        freeze_backbone_for_fairness(model, args.variant, args.train_lm_head)

    # LoRA (Track B) — identical on both variants
    if args.use_lora:
        model = add_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    # Optimizer param groups (τ-head lower LR; LoRA optional LR)
    param_groups = collect_param_groups(
        model,
        base_lr=args.lr,
        wd=args.weight_decay,
        variant=args.variant,
        lora_lr=(args.lora_lr if args.use_lora else None),
        tau_lr_mult=0.5,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    total_steps = max(1, len(train_loader) * args.epochs // max(1, args.grad_accum_steps))
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Train + early stop
    best_ppl = float("inf")
    bad = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.variant,
            lambda_reg=args.lambda_reg, grad_accum_steps=args.grad_accum_steps
        )
        ppl, ent, proxy, corr = evaluate(model, val_loader, device, args.variant)

        improved = ppl < best_ppl - 1e-6
        if improved:
            best_ppl = ppl; bad = 0
            save_path = Path(args.save_dir) / f"{args.variant}_best_ep{epoch}"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / "pytorch_model.bin")
            with open(save_path / "train_args.json", "w") as f:
                json.dump(vars(args), f, indent=2)
            tokenizer.save_pretrained(save_path.as_posix())
            with open(save_path / "README.txt", "w") as f:
                f.write(f"variant={args.variant} epoch={epoch} tag=best best_ppl={best_ppl}\n")
            print(f"[Checkpoint] Saved to {save_path}")
        else:
            bad += 1

        if args.save_every and (epoch % args.save_every == 0):
            save_dir = Path(args.save_dir) / f"{args.variant}_epoch_ep{epoch}"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(save_dir.as_posix())

        if args.variant == "thermal":
            print(f"[Epoch {epoch}] loss={loss:.4f} | ppl={ppl:.2f} | entropy={ent:.3f} | tau_mean={proxy:.5f} | corr(ent,tau)={corr:.3f}")
        else:
            print(f"[Epoch {epoch}] loss={loss:.4f} | ppl={ppl:.2f} | entropy={ent:.3f}")

        if args.early_stop_patience > 0 and bad >= args.early_stop_patience:
            print(f"[EarlyStop] no PPL improvement for {args.early_stop_patience} epoch(s).")
            break

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