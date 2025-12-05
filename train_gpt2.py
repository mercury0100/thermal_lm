#!/usr/bin/env python3
"""
Thermal LM Experiments (GPT-2) — Multi-dataset / Multi-variant
==============================================================

Variants:
    --variants baseline thermal

Datasets (all treated autoregressively, LM-style):
    - c4-small        → LM blocks from allenai/c4 (realnewslike)
    - wikitext-103    → LM blocks from wikitext-103-raw-v1
    - nq_open         → Natural Questions Open as "Q: ...\\nA: ..." LM text

This trains and evaluates models with:
  - Perplexity (LM-style next-token prediction)
  - Entropy, σ² statistics, corr(entropy, σ²) (for thermal/S-TLM variant)
  - Stepwise evaluation with --eval_every (more frequent than epochs)
  - Optional LoRA adapters with dropout (--use_lora)

Example:

python train_gpt2.py \
  --datasets c4-small wikitext-103 nq_open \
  --variants thermal \
  --freeze_backbone \
  --max_steps 20000
"""

import os
import math
import json
import csv
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import shutil  # for deleting old checkpoints

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

# Import the Thermal LM wrapper (from thermal_lm.py)
from thermal_lm import ThermalLMForCausalLM

# Quiet datasets/tokenizers
disable_progress_bar()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BAR", "1")


# ===============================
# Telemetry
# ===============================
class Telemetry:
    def __init__(self, out_dir: Path, use_wandb: bool = False, project: str = "thermal-gpt2"):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "train_log.csv"

        # OPEN IN WRITE MODE: start fresh each run
        self.csv_fp = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_fp)
        # Always write header
        self.writer.writerow(["time", "step", "epoch", "split", "metric", "value"])

        self.use_wandb = use_wandb
        self.wb = None
        if use_wandb:
            try:
                import wandb
                self.wb = wandb
                if not wandb.run:
                    wandb.init(project=project)
            except Exception:
                self.use_wandb = False

    def log(self, step: int, epoch: float, split: str, metrics: Dict[str, float]):
        ts = time.time()
        for k, v in metrics.items():
            self.writer.writerow([ts, step, epoch, split, k, v])
        self.csv_fp.flush()
        if self.use_wandb and self.wb is not None:
            self.wb.log({f"{split}/{k}": v for k, v in metrics.items()}, step=step)

    def close(self):
        try:
            self.csv_fp.close()
        except Exception:
            pass


# ===============================
# LoRA helpers
# ===============================
def try_import_peft():
    try:
        from peft import LoraConfig, get_peft_model
        return LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "PEFT is required when --use_lora is set. Install with `pip install peft`."
        ) from e


def add_lora(model, rank: int, alpha: int, dropout: float):
    """
    Wrap model with LoRA adapters on GPT-2-style projection modules.
    """
    LoraConfig, get_peft_model = try_import_peft()
    # GPT-2 blocks use c_attn (QKV together), c_fc, c_proj
    targets = ["c_attn", "c_fc", "c_proj"]
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    return get_peft_model(model, cfg)


def collect_param_groups(
    model,
    base_lr: float,
    wd: float,
    variant: str,
    lora_lr: Optional[float] = None,
    sigma_lr_mult: float = 0.5,
):
    """
    Separate base / thermal-head / LoRA params with possibly different learning rates.

    For the 'thermal' (S-TLM) variant, the thermal head is fc_logsigma.
    """
    groups = []
    lora_params, thermal_head_params, base_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n or "lora_A" in n or "lora_B" in n:
            lora_params.append(p)
        elif variant == "thermal" and n.startswith("fc_logsigma"):
            thermal_head_params.append(p)
        else:
            base_params.append(p)

    if base_params:
        groups.append({"params": base_params, "lr": base_lr, "weight_decay": wd})
    if thermal_head_params:
        groups.append({"params": thermal_head_params, "lr": base_lr * sigma_lr_mult, "weight_decay": wd})
    if lora_params:
        groups.append({"params": lora_params, "lr": lora_lr or base_lr, "weight_decay": wd})
    return groups


# ===============================
# Forward adapter
# ===============================
def forward_three(model, tokens, attn, variant: str):
    """
    Standardize outputs across variants.
    Returns: (logits_for_eval, logits_for_loss, logtemp_field)

      logits_*:      (B, T, V)
      logtemp_field: (B, T)   # here: log σ_t (or log τ_t for legacy)
    """
    if variant == "thermal":
        out = model(input_ids=tokens, attention_mask=attn)
        logits = out.logits
        aux = getattr(out, "aux", None)

        if aux is not None:
            if "logsigma" in aux:
                logtemp = aux["logsigma"]   # S-TLM head
            elif "logsigma" in aux:
                logtemp = aux["logsigma"]     # legacy thermal head, if present
            else:
                B, T, _ = logits.size()
                logtemp = torch.zeros(B, T, device=logits.device, dtype=logits.dtype)
        else:
            B, T, _ = logits.size()
            logtemp = torch.zeros(B, T, device=logits.device, dtype=logits.dtype)

        return logits, logits, logtemp
    else:
        out = model(input_ids=tokens, attention_mask=attn)
        logits = out.logits
        B, T, _ = logits.size()
        zeros = torch.zeros(B, T, device=logits.device, dtype=logits.dtype)
        return logits, logits, zeros


# ===============================
# Datasets (autoregressive LM)
# ===============================
def lm_build_train_val(tokenizer, name: str, block_size: int, batch_size: int):
    """C4 / C4-small (streaming) → fixed-length LM blocks without padding."""
    from itertools import islice
    name = name.lower()
    assert name in ["c4", "c4-small"], f"Unsupported dataset for LM: {name}"
    if name == "c4":
        train_blocks, val_blocks = 50000, 2000
    else:
        train_blocks, val_blocks = 10000, 1000

    ds = load_dataset("allenai/c4", "realnewslike", streaming=True)
    eos = tokenizer.eos_token_id or 0

    def stream_blocks(example_iter, n_blocks):
        buf = []
        yielded = 0
        for ex in example_iter:
            text = ex.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=True)
            if eos:
                ids.append(eos)
            buf.extend(ids)
            while len(buf) >= block_size and yielded < n_blocks:
                block = buf[:block_size]
                buf = buf[block_size:]
                yielded += 1
                yield torch.tensor(block, dtype=torch.long)
                if yielded >= n_blocks:
                    break

    tr_it = ds["train"].shuffle(seed=42)
    va_it = ds["validation"].shuffle(seed=43)
    train_ids = list(islice(stream_blocks(tr_it, train_blocks), train_blocks))
    val_ids = list(islice(stream_blocks(va_it, val_blocks), val_blocks))

    train = TensorDataset(
        torch.stack(train_ids),
        torch.ones_like(train_ids[0]).repeat(len(train_ids), 1),
    )
    val = TensorDataset(
        torch.stack(val_ids),
        torch.ones_like(val_ids[0]).repeat(len(val_ids), 1),
    )

    def collate(b):
        x = torch.stack([t[0] for t in b])
        m = torch.stack([t[1] for t in b])
        return {"input_ids": x, "attention_mask": m}

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate),
        DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate),
    )


def wikitext103_build_train_val(tokenizer, block_size: int, batch_size: int):
    """wikitext-103-raw-v1 → fixed-length LM blocks."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=True)

    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        if total_len == 0:
            return {"input_ids": [], "attention_mask": []}
        ids = [concatenated[i:i + block_size] for i in range(0, total_len, block_size)]
        attn = [[1] * block_size for _ in range(len(ids))]
        return {"input_ids": ids, "attention_mask": attn}

    train_ds = ds["train"].map(tokenize_fn, batched=True, remove_columns=["text"]).map(
        group_texts, batched=True
    )
    val_ds = ds["validation"].map(tokenize_fn, batched=True, remove_columns=["text"]).map(
        group_texts, batched=True
    )

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def nq_open_build_train_val(
    tokenizer,
    block_size: int,
    batch_size: int,
    max_train_examples: Optional[int] = None,
    val_frac: float = 0.05,
):
    """
    nq_open → LM over 'Q: ...\\nA: ...' strings.
    Perplexity is plain next-token LM; no special labels.
    """
    ds = load_dataset("google-research-datasets/nq_open")
    raw = ds["train"]

    if max_train_examples is not None and max_train_examples > 0 and len(raw) > max_train_examples:
        raw = raw.shuffle(seed=123).select(range(max_train_examples))

    split = raw.train_test_split(test_size=val_frac, seed=321)
    train_raw, val_raw = split["train"], split["test"]

    def qa_to_text(ex):
        q = ex.get("question", "")
        ans = ex.get("answer", [])
        if isinstance(ans, list) and len(ans) > 0:
            a = ans[0]
        else:
            a = str(ans) if ans is not None else ""
        text = f"Q: {q}\nA: {a}"
        return {"text": text}

    train_raw = train_raw.map(qa_to_text)
    val_raw = val_raw.map(qa_to_text)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=True)

    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        if total_len == 0:
            return {"input_ids": [], "attention_mask": []}
        ids = [concatenated[i:i + block_size] for i in range(0, total_len, block_size)]
        attn = [[1] * block_size for _ in range(len(ids))]
        return {"input_ids": ids, "attention_mask": attn}

    # Important: drop all original columns (question, answer, etc.) before grouping
    train_ds = (
        train_raw
        .map(tokenize_fn, batched=True, remove_columns=train_raw.column_names)
        .map(group_texts, batched=True)
    )
    val_ds = (
        val_raw
        .map(tokenize_fn, batched=True, remove_columns=val_raw.column_names)
        .map(group_texts, batched=True)
    )

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# ===============================
# Evaluation (PPL, entropy, σ² corr)
# ===============================
@torch.no_grad()
@torch.no_grad()
def evaluate_autoregressive(model, loader, device, variant: str):
    """
    LM-style evaluation:
      - labels are next-token targets (shifted input_ids)
      - perplexity over all non-masked tokens

    For the thermal variant, we also compute:
      - mean token entropy
      - mean σ² (treated as local temperature)
      - corr(CE, σ²): correlation between per-token cross-entropy and σ²
    """
    model.eval()
    total_nll = 0.0
    total_tok = 0

    # For mean entropy / mean temperature
    ent_sum = 0.0
    temp_sum = 0.0
    temp_count = 0

    # For correlation between CE and σ²
    n_ce = 0
    s_ce = s_temp = s_ce2 = s_temp2 = s_ce_temp = 0.0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        tokens = ids[:, :-1].contiguous()
        labels_tgt = ids[:, 1:].contiguous()
        attn_tok = attn[:, :-1].contiguous()
        valid = attn_tok > 0  # (B, T)

        logits_eval, logits_loss, logtemp = forward_three(model, tokens, attn_tok, variant)
        B, T, V = logits_loss.size()

        labels_flat = labels_tgt.view(-1)
        logits_flat = logits_loss.view(-1, V)
        valid_flat = valid.view(-1)

        # Per-token CE (flattened)
        ce_all = F.cross_entropy(
            logits_flat,
            labels_flat.clamp_min(0),
            reduction="none",
        )
        ce_valid = ce_all[valid_flat]
        total_nll += float(ce_valid.sum().item())
        total_tok += int(valid_flat.sum().item())

        # Entropy & temperature stats on valid positions
        probs = torch.softmax(logits_eval, dim=-1)
        token_ent_all = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)  # (B, T)

        if variant == "thermal":
            # logtemp is log σ_t; treat σ_t² as micro-temperature
            sigma = logtemp.exp()
            temp_field_all = sigma ** 2  # (B, T)
        else:
            temp_field_all = torch.zeros_like(token_ent_all)

        # ---- Mean entropy / mean temperature over valid tokens ----
        ent_valid = token_ent_all[valid].double()      # (N_valid_batch,)
        temp_valid = temp_field_all[valid].double()    # (N_valid_batch,)

        nb = ent_valid.numel()
        if nb > 0:
            ent_sum += ent_valid.sum().item()
            temp_sum += temp_valid.sum().item()
            temp_count += nb

        # ---- Correlation between CE and temperature (σ²) ----
        # reshape CE back to (B, T) to align with temp_field_all
        ce_all_tok = ce_all.view(B, T)                 # (B, T)
        ce_valid_tok = ce_all_tok[valid].double()      # (N_valid_batch,)

        # temp_valid already aligned (same mask)
        ce_vec = ce_valid_tok
        temp_vec = temp_valid

        nb_ce = ce_vec.numel()
        if nb_ce > 0:
            n_ce += nb_ce
            s_ce += ce_vec.sum().item()
            s_temp += temp_vec.sum().item()
            s_ce2 += (ce_vec * ce_vec).sum().item()
            s_temp2 += (temp_vec * temp_vec).sum().item()
            s_ce_temp += (ce_vec * temp_vec).sum().item()

    # Perplexity
    if total_tok == 0:
        ppl = float("inf")
    else:
        mean_nll = total_nll / total_tok
        ppl = math.exp(mean_nll)

    # Means
    if temp_count > 0:
        ent_mean = ent_sum / temp_count
        temp_mean = temp_sum / temp_count
    else:
        ent_mean = 0.0
        temp_mean = 0.0

    # Corr(CE, σ²)
    denom_ce = n_ce * s_ce2 - (s_ce ** 2)
    denom_temp = n_ce * s_temp2 - (s_temp ** 2)
    if n_ce > 1 and denom_ce > 0 and denom_temp > 0:
        corr_ce_temp = (n_ce * s_ce_temp - s_ce * s_temp) / math.sqrt(denom_ce * denom_temp)
    else:
        corr_ce_temp = 0.0

    return float(ppl), float(ent_mean), float(temp_mean), float(corr_ce_temp)


# ===============================
# Training step
# ===============================
def train_step(model, batch, device):
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    # LM-style: labels = ids (ThermalLM/GPT2 will shift internally)
    out = model(input_ids=ids, attention_mask=attn, labels=ids)
    return out.loss


# ===============================
# One experiment (dataset, variant)
# ===============================
def train_one_task(args):
    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Data (all LM-style)
    if args.task != "lm":
        raise ValueError(f"Unsupported task: {args.task} (only 'lm' is used)")

    ds_name = args.train_dataset.lower()
    if ds_name in {"c4-small", "c4"}:
        train_loader, val_loader = lm_build_train_val(
            tokenizer,
            "c4-small" if ds_name == "c4-small" else "c4",
            args.max_length,
            args.batch_size,
        )
    elif ds_name in {"wikitext-103", "wikitext103"}:
        train_loader, val_loader = wikitext103_build_train_val(
            tokenizer,
            args.max_length,
            args.batch_size,
        )
    elif ds_name in {"nq_open", "natural-questions", "naturalquestions"}:
        train_loader, val_loader = nq_open_build_train_val(
            tokenizer,
            args.max_length,
            args.batch_size,
        )
    else:
        raise ValueError(f"Unknown LM dataset: {args.train_dataset}")

    metric_name = "ppl"
    higher_is_better = False

    # Model
    if args.variant == "thermal":
        # Map sigma_* CLI flags onto logsigma_* / sigma_smooth config for S-TLM
        model = ThermalLMForCausalLM.from_base_pretrained(
            base_model_name_or_path=args.base_model,
            logsigma_clamp_min=args.sigma_clamp_min,
            logsigma_clamp_max=args.sigma_clamp_max,
            lambda_reg=args.lambda_reg,
            sigma_smooth=args.sigma_smooth,
            noise_at_eval=False,  # deterministic eval by default
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)

    # Optional LoRA
    if args.use_lora:
        model = add_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    # optionally freeze backbone
    if args.freeze_backbone:
        if args.variant == "thermal":
            # Only train the diffusion head (fc_logsigma*)
            for n, p in model.named_parameters():
                if n.startswith("fc_logsigma"):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            # Baseline: everything frozen
            for p in model.parameters():
                p.requires_grad = False

    # Telemetry / checkpoints
    dataset_tag = args.train_dataset
    ckpt_tag = f"{args.variant}_{dataset_tag}_gpt2"
    out_dir = Path(args.save_dir) / ckpt_tag

    # Wipe previous runs for this (variant, dataset)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tel = Telemetry(out_dir, use_wandb=args.wandb, project="thermal-gpt2")

    # Initial eval
    model.eval()
    ppl_init, ent_init, temp_init, corr_init = evaluate_autoregressive(
        model, val_loader, device, args.variant
    )
    tel.log(0, 0.0, "val_init", {
        "ppl": ppl_init,
        "entropy": ent_init,
        "sigma_mean": temp_init,          # logged as sigma_mean for backward-compat
        "corr_ce_sigma": corr_init,
    })
    print(
        f"[Init Eval] PPL={ppl_init:.4f} | Entropy={ent_init:.4f} | "
        f"SigmaMean={temp_init:.4f} | Corr(CE,σ²)={corr_init:.4f}"
    )

    # if baseline + freeze_backbone, just checkpoint and exit
    if args.freeze_backbone and args.variant == "baseline":
        save_path = out_dir / f"frozen_init_ppl{ppl_init:.4f}"
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / "pytorch_model.bin")
        tokenizer.save_pretrained(save_path.as_posix())
        with open(save_path / "train_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[FreezeBackbone] Baseline frozen checkpoint saved at: {save_path}")
        tel.close()
        return

    best_metric = ppl_init
    best_path: Optional[Path] = None
    global_step = 0
    bad_steps = 0
    running_loss = 0.0

    # Optimizer: param groups for base / thermal head / LoRA (after freezing)
    param_groups = collect_param_groups(
        model,
        base_lr=args.lr,
        wd=args.weight_decay,
        variant=args.variant,
        lora_lr=args.lora_lr if args.use_lora else None,
        sigma_lr_mult=args.sigma_lr_mult,
    )
    if len(param_groups) == 0:
        print("[Warning] No trainable parameters found. Exiting.")
        tel.close()
        return

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    steps_per_epoch = len(train_loader)
    nominal_total_steps = max(1, steps_per_epoch * args.epochs)
    total_steps = (
        nominal_total_steps if args.max_steps <= 0 else min(nominal_total_steps, args.max_steps)
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        max(1, int(0.1 * total_steps)),
        total_steps,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        for i, batch in enumerate(train_loader):
            loss = train_step(model, batch, device)
            (loss / args.grad_accum_steps).backward()

            update_now = ((i + 1) % args.grad_accum_steps == 0) or ((i + 1) == len(train_loader))
            if update_now:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running_loss += float(loss.item())

                # Logging
                if global_step % args.log_every == 0:
                    avg_loss = running_loss / max(1, args.log_every)
                    tel.log(global_step, epoch + i / steps_per_epoch, "train", {"loss": avg_loss})
                    running_loss = 0.0

                # Eval
                if global_step % args.eval_every == 0:
                    model.eval()
                    ppl, ent, temp_mean, corr = evaluate_autoregressive(
                        model, val_loader, device, args.variant
                    )
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Eval step {global_step:05d} | "
                        f"epoch {epoch:.2f} | PPL={ppl:.4f} | "
                        f"Entropy={ent:.4f} | SigmaMean={temp_mean:.4f} | Corr(CE,σ²)={corr:.4f}"
                    )
                    tel.log(global_step, epoch + i / steps_per_epoch, "val", {
                        "ppl": ppl,
                        "entropy": ent,
                        "sigma_mean": temp_mean,
                        "corr_ce_sigma": corr,
                    })

                    improved = (ppl < best_metric) if not higher_is_better else (ppl > best_metric)
                    if improved:
                        best_metric = ppl
                        bad_steps = 0

                        # delete previous best checkpoint (if any)
                        if best_path is not None and best_path.exists():
                            shutil.rmtree(best_path)

                        save_path = out_dir / f"best_step{global_step}_ppl{best_metric:.4f}"
                        save_path.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), save_path / "pytorch_model.bin")
                        tokenizer.save_pretrained(save_path.as_posix())
                        with open(save_path / "train_args.json", "w") as f:
                            json.dump(vars(args), f, indent=2)
                        best_path = save_path
                        print(f"[Checkpoint] PPL improved → {best_metric:.4f} at step {global_step}")
                    else:
                        bad_steps += args.eval_every
                        if args.patience_steps > 0 and bad_steps >= args.patience_steps:
                            print(
                                f"[EarlyStop] no improvement for {bad_steps} steps. "
                                f"Best at: {best_path}"
                            )
                            tel.close()
                            return
                    model.train()

                # Max steps override
                if args.max_steps > 0 and global_step >= args.max_steps:
                    print(
                        f"[MaxSteps] Reached max_steps={args.max_steps}. "
                        f"Stopping. Best at: {best_path}"
                    )
                    tel.close()
                    return

    tel.close()


# ===============================
# Orchestration (multi-dataset / multi-variant)
# ===============================
def resolve_task_and_dataset(name: str) -> Tuple[str, Optional[str]]:
    """Maps a dataset token to (task, train_dataset_tag_for_lm)."""
    key = name.lower()
    if key in {"c4-small", "c4"}:
        return "lm", ("c4-small" if key == "c4-small" else "c4")
    if key in {"wikitext-103", "wikitext103"}:
        return "lm", "wikitext-103"
    if key in {"nq_open", "natural-questions", "naturalquestions"}:
        return "lm", "nq_open"
    raise ValueError(f"Unknown dataset token: {name}")


def run_experiment(args, dataset: str, variant: str):
    # Clone args to avoid mutation across runs
    local = argparse.Namespace(**vars(args))
    task, lm_name = resolve_task_and_dataset(dataset)
    local.task = task
    if task == "lm":
        local.train_dataset = lm_name
    local.variant = variant
    print(
        f"\n[RUN] variant={variant} dataset={dataset} → task={local.task} "
        f"{'(train_dataset='+lm_name+')' if lm_name else ''}"
    )
    train_one_task(local)


def main():
    p = argparse.ArgumentParser()

    # Multi-selection
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["c4-small", "wikitext-103", "nq_open"],
        help="Datasets: c4-small, c4, wikitext-103, nq_open",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=["thermal"],
        choices=["baseline", "thermal"],
        help="Model variants to run.",
    )

    # Core
    p.add_argument("--variant", type=str, default="thermal", choices=["baseline", "thermal"])
    p.add_argument("--base_model", type=str, default="gpt2")
    p.add_argument(
        "--task",
        type=str,
        default="lm",
        choices=["lm"],
        help="Internal use; all datasets are LM-style.",
    )
    p.add_argument(
        "--train_dataset",
        type=str,
        default="c4-small",
        help="For task=lm: one of c4-small, c4, wikitext-103, nq_open",
    )

    # Train hparams
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--grad_accum_steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If >0, stop after this many optimizer update steps (global steps).",
    )

    # Thermal knobs (λ for logσ² prior; CLI keeps sigma_* names for now)
    p.add_argument("--lambda_reg", type=float, default=0., help="λ for (logσ)²")
    p.add_argument("--sigma_clamp_min", type=float, default=-6.0)
    p.add_argument("--sigma_clamp_max", type=float, default=2.0)
    p.add_argument("--sigma_smooth", action="store_true")
    p.add_argument(
        "--sigma_lr_mult",
        type=float,
        default=0.5,
        help="LR multiplier for thermal head (σ-head in S-TLM).",
    )

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_lr", type=float, default=1e-4)

    # Freeze backbone
    p.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="If set: baseline will only be evaluated & checkpointed; thermal will train only fc_logsigma.",
    )

    # Saving / telemetry / eval cadence
    p.add_argument("--save_dir", type=str, default="checkpoints_gpt2")
    p.add_argument("--eval_every", type=int, default=50, help="steps between evals")
    p.add_argument("--log_every", type=int, default=50, help="steps between telemetry logs")
    p.add_argument(
        "--patience_steps",
        type=int,
        default=300,
        help="early stop if no improvement for this many steps (0=off)",
    )
    p.add_argument("--wandb", action="store_true")

    args = p.parse_args()

    # Single-run fallback
    if args.datasets == [] and args.variants == []:
        train_one_task(args)
        return

    # Loop over all combinations
    for variant in args.variants:
        for dataset in args.datasets:
            run_experiment(args, dataset, variant)


if __name__ == "__main__":
    main()
