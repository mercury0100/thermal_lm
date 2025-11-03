#!/usr/bin/env python3
"""
Thermal LM Experiments (GPT-2) — Baseline vs Thermal (Free-Energy)
-------------------------------------------------------------------

Variants:
    --variant baseline   → Standard GPT-2 fine-tune
    --variant thermal    → Thermal LM with per-token learned temperature τ(x)

This trains and evaluates models with:
  - Perplexity, entropy, τ statistics, and corr(entropy, τ)
  - Calibration (ECE + reliability diagrams)
  - Optional epistemic MI via dropout
  - Optional OOD perplexity evaluation
"""

import os
import math
import json
import csv
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

# Import your Thermal LM wrapper
from thermal_lm import ThermalLMForCausalLM

# Disable Hugging Face progress noise
disable_progress_bar()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BAR", "1")


# ---------------------------------------------------------------
# Forward Adapter
# ---------------------------------------------------------------
def forward_three(model, tokens, attn, variant: str):
    """
    Standardize outputs across variants.
    Returns: (logits, logits_for_loss, aux_logtau)
    """
    if variant == "thermal":
        out = model(input_ids=tokens, attention_mask=attn)
        logits = out.logits
        aux = getattr(out, "aux", None)
        logtau = aux.get("logtau") if aux and "logtau" in aux else torch.zeros_like(tokens, dtype=logits.dtype)
        return logits, logits, logtau
    else:
        out = model(input_ids=tokens, attention_mask=attn)
        logits = out.logits
        B, T, _ = logits.size()
        zeros = torch.zeros(B, T, device=logits.device, dtype=logits.dtype)
        return logits, logits, zeros

# ---------------------------------------------------------------
# Evaluation (PPL, entropy, τ correlation)
# ---------------------------------------------------------------
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
        attn = attention_mask[:, :-1].contiguous()

        _, logits, aux = forward_three(model, tokens, attn, variant)
        B, T, V = logits.size()

        nll = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += B * T

        # Entropy from base logits (no τ)
        if variant == "thermal":
            if hasattr(model, "transformer"):
                h = model.transformer(input_ids=tokens, attention_mask=attn).last_hidden_state
                logits_base = model.lm_head(h)
            else:
                logits_base = model.base(tokens).logits
            probs = torch.softmax(logits_base, dim=-1)
            token_ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            tau_field = aux.exp()
        else:
            probs = torch.softmax(logits, dim=-1)
            token_ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            tau_field = torch.zeros_like(token_ent)

        x = token_ent.reshape(-1).double()
        y = tau_field.reshape(-1).double()
        nb = x.numel()
        n += nb
        sx += x.sum().item()
        sy += y.sum().item()
        sx2 += (x * x).sum().item()
        sy2 += (y * y).sum().item()
        sxy += (x * y).sum().item()

    ppl = math.exp(total_nll / max(1, total_tokens))
    denom_x = n * sx2 - (sx ** 2)
    denom_y = n * sy2 - (sy ** 2)
    corr = (n * sxy - sx * sy) / math.sqrt(max(denom_x, 1e-12) * max(denom_y, 1e-12)) if (n > 1 and denom_x > 0 and denom_y > 0) else 0.0

    ent_mean = sx / max(1, n)
    tau_mean = sy / max(1, n)
    return ppl, float(ent_mean), float(tau_mean), float(corr)


# ---------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------
def build_dataloaders(tokenizer, block_size, batch_size):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=True)

    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        input_ids = [concatenated[i:i + block_size] for i in range(0, total_len, block_size)]
        attention_mask = [[1] * block_size for _ in range(len(input_ids))]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    train_ds = ds["train"].map(tokenize_fn, batched=True, remove_columns=["text"]).map(group_texts, batched=True)
    val_ds = ds["validation"].map(tokenize_fn, batched=True, remove_columns=["text"]).map(group_texts, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, variant, grad_accum_steps=1):
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
            out = model(input_ids=tokens, attention_mask=attn, labels=labels)
            loss = out.loss / grad_accum_steps
        else:
            logits = model(input_ids=tokens, attention_mask=attn).logits
            B, T, V = logits.size()
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                labels.reshape(-1),
                reduction="mean"
            ) / grad_accum_steps

        loss.backward()

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps
        steps += 1

    return total_loss / max(1, steps)

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=["baseline", "thermal"], default="thermal")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lambda_reg", type=float, default=1e-3, help="λ regularizer for logτ²")
    parser.add_argument("--alpha_entropy", type=float, default=1e-3, help="α coefficient for entropy term")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Running {args.variant.upper()} variant on {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_loader, val_loader = build_dataloaders(tokenizer, args.max_length, args.batch_size)

    if args.variant == "thermal":
        model = ThermalLMForCausalLM.from_base_pretrained(
            "gpt2",
            lambda_reg=args.lambda_reg,
            alpha_entropy=args.alpha_entropy
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or "ln" in n.lower() or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01, "lr": args.lr},
         {"params": no_decay, "weight_decay": 0.0, "lr": args.lr}]
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_ppl = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.variant,
                           grad_accum_steps=args.grad_accum_steps)
        ppl, ent, tau_mean, corr = evaluate(model, val_loader, device, args.variant)
        if ppl < best_ppl:
            best_ppl = ppl
            Path(args.save_dir).mkdir(exist_ok=True)
            torch.save(model.state_dict(), Path(args.save_dir) / f"{args.variant}_best_ep{epoch}.pt")
            print(f"[Checkpoint] Saved best model at epoch {epoch}")
        print(f"[Epoch {epoch}] FreeEnergy={loss:.4f} | PPL={ppl:.2f} | "
              f"Entropy={ent:.3f} | TauMean={tau_mean:.4f} | Corr(ent,tau)={corr:.3f}")


if __name__ == "__main__":
    main()
