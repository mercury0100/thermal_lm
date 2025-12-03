#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermal LM Experiments — LLaMA family (multi-dataset / multi-variant)
=====================================================================

What this module does
---------------------
- Fine-tunes **baseline** (vanilla causal LM) and/or **thermal** (your TLM wrapper)
  on one or more **autoregressive** datasets in one go.
- Turnkey support for: **c4-small** (LM), **triviaqa** (generative QA),
  **truthfulqa** (generative QA).
- Keeps your original training loop, telemetry, eval cadence, and checkpointing.
- Adds simple orchestration flags:
    --datasets  c4-small triviaqa truthfulqa
    --variants  baseline thermal

Requirements
------------
- torch, transformers, datasets, numpy
- (optional) peft (for LoRA) and wandb (for telemetry)
- Your module `ThermalLMForCausalLM` importable as `from thermal_lm import ThermalLMForCausalLM`

python train_llama.py \
 --base_model meta-llama/Llama-3.2-1B \
  --datasets c4-small triviaqa truthfulqa \
  --variants baseline thermal \
  --epochs 2 \
  --batch_size 2 \
  --lr 2e-4 \
  --grad_accum_steps 32 \
  --max_length 1024 \
  --use_lora --freeze_backbone --train_lm_head bias
"""

import os
import math
import json
import time
import csv
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from datasets import load_dataset, disable_progress_bar, concatenate_datasets
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

# ===============================
# Telemetry
# ===============================
class Telemetry:
    def __init__(self, out_dir: Path, use_wandb: bool = False, project: str = "thermal-llama"):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "train_log.csv"
        self.csv_fp = open(self.csv_path, "a", newline="")
        self.writer = csv.writer(self.csv_fp)
        if self.csv_path.stat().st_size == 0:
            self.writer.writerow(["time", "step", "epoch", "split", "metric", "value"])  # header
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
# LoRA / freezing helpers
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
    LoraConfig, get_peft_model = try_import_peft()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    )
    return get_peft_model(model, cfg)


def freeze_backbone_for_fairness(model, variant: str, train_lm_head: str):
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
                         variant: str, lora_lr: Optional[float] = None, tau_lr_mult: float = 0.5):
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

# ===============================
# Tasks & data (autoregressive only)
# ===============================

def lm_build_train_val(tokenizer, name: str, block_size: int, batch_size: int):
    """C4 / C4-small streaming → fixed-length blocks without padding."""
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
                block = buf[:block_size]; buf = buf[block_size:]
                yielded += 1
                yield torch.tensor(block, dtype=torch.long)
                if yielded >= n_blocks:
                    break
    tr_it = ds["train"].shuffle(seed=42)
    va_it = ds["validation"].shuffle(seed=43)
    train_ids = list(islice(stream_blocks(tr_it, train_blocks), train_blocks))
    val_ids   = list(islice(stream_blocks(va_it, val_blocks),   val_blocks))
    train = TensorDataset(torch.stack(train_ids), torch.ones_like(train_ids[0]).repeat(len(train_ids),1))
    val   = TensorDataset(torch.stack(val_ids),   torch.ones_like(val_ids[0]).repeat(len(val_ids),1))
    def collate(b):
        x = torch.stack([t[0] for t in b]); m = torch.stack([t[1] for t in b]); return {"input_ids": x, "attention_mask": m}
    return DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate), \
           DataLoader(val,   batch_size=batch_size, shuffle=False, collate_fn=collate)


def triviaqa_build(tokenizer, batch_size: int, split="train", max_length=1024, answer_maxlen=128):
    ds = load_dataset("trivia_qa", "unfiltered")
    data = ds[split]
    prompt_cache = {}
    def prompt_ids(prompt: str):
        if prompt not in prompt_cache:
            prompt_cache[prompt] = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
        return prompt_cache[prompt]

    def to_pair(ex):
        q = ex["question"]
        a = ex["answer"]["value"] if isinstance(ex.get("answer"), dict) else (ex.get("answer") or "")
        a = a.strip().split("; ")[0]
        prompt = f"Q: {q}\nA: "
        pids = prompt_ids(prompt)
        ans_ids = tokenizer(" " + a, add_special_tokens=False, truncation=True, max_length=answer_maxlen)["input_ids"]
        ids = torch.tensor((pids + ans_ids)[:max_length], dtype=torch.long)
        labels = torch.full_like(ids, -100)
        cut = min(len(pids), len(ids))
        labels[cut:len(ids)] = ids[cut:len(ids)]
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}

    proc = data.map(lambda ex: to_pair(ex), remove_columns=data.column_names)
    proc.set_format(type="torch")
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        ids = []; attn = []; labels = []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            ids.append(torch.cat([b["input_ids"], torch.full((pad,), tokenizer.pad_token_id)], 0))
            attn.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)], 0))
            lab = torch.full((maxlen,), -100)
            valid = b["labels"] != -100
            lab[-valid.sum():] = b["labels"][valid]
            labels.append(lab)
        return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(attn), "labels": torch.stack(labels)}
    return DataLoader(proc, batch_size=batch_size, shuffle=True, collate_fn=collate), \
           DataLoader(proc, batch_size=batch_size, shuffle=False, collate_fn=collate)


def truthfulqa_build(tokenizer, batch_size: int, split="validation", max_length=1024, answer_maxlen=128):
    ds = load_dataset("truthful_qa", "generation")
    data = ds[split]
    prompt_cache = {}
    def prompt_ids(prompt: str):
        if prompt not in prompt_cache:
            prompt_cache[prompt] = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
        return prompt_cache[prompt]

    def to_pair(ex):
        q = ex["question"]
        best = ex.get("best_answer") or (ex.get("best_answer_list") or [""])[0]
        prompt = f"Q: {q}\nA: "
        pids = prompt_ids(prompt)
        ans_ids = tokenizer(" " + best, add_special_tokens=False, truncation=True, max_length=answer_maxlen)["input_ids"]
        ids = torch.tensor((pids + ans_ids)[:max_length], dtype=torch.long)
        labels = torch.full_like(ids, -100)
        cut = min(len(pids), len(ids))
        labels[cut:len(ids)] = ids[cut:len(ids)]
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}

    proc = data.map(lambda ex: to_pair(ex), remove_columns=data.column_names)
    proc.set_format(type="torch")
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        ids = []; attn = []; labels = []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            ids.append(torch.cat([b["input_ids"], torch.full((pad,), tokenizer.pad_token_id)], 0))
            attn.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)], 0))
            lab = torch.full((maxlen,), -100)
            valid = b["labels"] != -100
            lab[-valid.sum():] = b["labels"][valid]
            labels.append(lab)
        return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(attn), "labels": torch.stack(labels)}
    return DataLoader(proc, batch_size=batch_size, shuffle=True, collate_fn=collate), \
           DataLoader(proc, batch_size=batch_size, shuffle=False, collate_fn=collate)

# ===============================
# Evaluation helpers (autoregressive)
# ===============================
@torch.no_grad()
def evaluate_lm_ppl(model, loader, device) -> float:
    model.eval()
    total_loss, total_batches = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn= batch["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=attn, labels=ids)
        total_loss += float(out.loss.item())
        total_batches += 1
    mean_loss = total_loss / max(1, total_batches)
    return float(math.exp(mean_loss))


@torch.no_grad()
def evaluate_qa_ppl(model, loader, device) -> float:
    """Per-token perplexity over supervised answer tokens (labels != -100)."""
    model.eval()
    total_nll = 0.0
    total_tok = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=ids, attention_mask=attn, labels=labels)
        loss = out.loss
        n_tok = (labels != -100).sum().item()
        if n_tok == 0:
            continue
        total_nll += float(loss.item()) * n_tok
        total_tok += n_tok
    if total_tok == 0:
        return float("inf")
    mean_nll = total_nll / total_tok
    return float(math.exp(mean_nll))

# ===============================
# Training step
# ===============================

def train_step(model, batch, device, variant):
    ids = batch["input_ids"].to(device)
    attn= batch["attention_mask"].to(device)
    labels = batch.get("labels")
    if labels is not None:
        labels = labels.to(device)
    out = model(input_ids=ids, attention_mask=attn, labels=ids if labels is None else labels)
    return out.loss

# ===============================
# One experiment (dataset, variant)
# ===============================

def train_one_task(args):
    # Repro
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Data by task (autoregressive only)
    if args.task == "lm":
        train_loader, val_loader = lm_build_train_val(tokenizer, args.train_dataset, args.max_length, args.batch_size)
        metric_name = "ppl"; higher_is_better = False
    elif args.task == "triviaqa":
        train_loader, val_loader = triviaqa_build(tokenizer, args.batch_size, split="train", max_length=args.max_length)
        metric_name = "ppl"; higher_is_better = False
    elif args.task == "truthfulqa":
        train_loader, val_loader = truthfulqa_build(tokenizer, args.batch_size, split="validation", max_length=args.max_length)
        metric_name = "ppl"; higher_is_better = False
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # Model
    if args.variant == "thermal":
        model = ThermalLMForCausalLM.from_base_pretrained(
            args.base_model,
            tau_clamp_min=args.tau_clamp_min,
            tau_clamp_max=args.tau_clamp_max,
            lambda_reg=args.lambda_reg,
            alpha_entropy=args.alpha_entropy,
            tau_smooth=args.tau_smooth,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)

    # Optionally freeze or LoRA
    if args.freeze_backbone:
        freeze_backbone_for_fairness(model, args.variant, args.train_lm_head)
    if args.use_lora:
        model = add_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    # Optimizer/scheduler
    param_groups = collect_param_groups(model, args.lr, args.weight_decay, args.variant, args.lora_lr if args.use_lora else None)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1,int(0.1*total_steps)), total_steps)

    # Telemetry / checkpoint dir
    dataset_tag = args.train_dataset if args.task == "lm" else args.task
    ckpt_tag = f"{args.variant}_{dataset_tag}_sft"
    out_dir = Path(args.save_dir) / ckpt_tag
    tel = Telemetry(out_dir, use_wandb=args.wandb, project="thermal-llama")

    # Training loop (step-wise eval & early stop)
    # --- before training loop ---
    model.eval()
    if args.task == "lm":
        metric = evaluate_lm_ppl(model, val_loader, device)
    else:
        metric = evaluate_qa_ppl(model, val_loader, device)
    tel.log(0, 0.0, "val_init", {metric_name: metric})
    print(f"[Init Eval] {metric_name}: {metric:.4f}")
    best_metric = metric
    best_path = None
    global_step = 0
    bad_steps = 0
    running = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for i, batch in enumerate(train_loader):
            loss = train_step(model, batch, device, args.variant)
            (loss / args.grad_accum_steps).backward()
            if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
                running += float(loss.item())

                if global_step % args.log_every == 0:
                    tel.log(global_step, epoch + i/steps_per_epoch, "train", {"loss": running/max(1,args.log_every)})
                    running = 0.0

                if global_step % args.eval_every == 0:
                    # Eval
                    model.eval()
                    if args.task == "lm":
                        metric = evaluate_lm_ppl(model, val_loader, device)
                    else:
                        metric = evaluate_qa_ppl(model, val_loader, device)
                    print(f"[{time.strftime('%H:%M:%S')}] Eval step {global_step:05d} | epoch {epoch:.2f} | "
                          f"{metric_name}: {metric:.4f} ({'↑' if higher_is_better else '↓'} better)")
                    tel.log(global_step, epoch + i/steps_per_epoch, "val", {metric_name: metric})

                    improved = (metric > best_metric) if higher_is_better else (metric < best_metric)
                    if improved:
                        best_metric = metric; bad_steps = 0
                        save_path = out_dir / f"best_step{global_step}_{metric_name}{best_metric:.4f}"
                        save_path.mkdir(parents=True, exist_ok=True)
                        # save state dict + tokenizer + args
                        torch.save(model.state_dict(), save_path / "pytorch_model.bin")
                        tokenizer.save_pretrained(save_path.as_posix())
                        with open(save_path / "train_args.json", "w") as f:
                            json.dump(vars(args), f, indent=2)
                        best_path = save_path
                        print(f"[Checkpoint] {metric_name} improved → {best_metric:.4f} at step {global_step}")
                    else:
                        bad_steps += args.eval_every
                        if args.patience_steps > 0 and bad_steps >= args.patience_steps:
                            print(f"[EarlyStop] no improvement for {bad_steps} steps. Best at: {best_path}")
                            tel.close(); return
                    model.train()  # back to train

    tel.close()

# ===============================
# Orchestration (multi-dataset / multi-variant)
# ===============================

def resolve_task_and_dataset(name: str):
    """Maps a dataset token to (task, train_dataset_tag_for_lm)."""
    key = name.lower()
    if key in {"c4-small", "c4"}:
        return ("lm", "c4-small" if key == "c4-small" else "c4")
    if key in {"triviaqa", "truthfulqa"}:
        return (key, None)
    raise ValueError(f"Unknown dataset token: {name}")


def run_experiment(args, dataset: str, variant: str):
    # Clone args to avoid mutation across runs
    local = argparse.Namespace(**vars(args))
    task, lm_name = resolve_task_and_dataset(dataset)
    local.task = "lm" if task == "lm" else task
    if task == "lm":
        local.train_dataset = lm_name
    local.variant = variant
    print(f"\n[RUN] variant={variant} dataset={dataset} → task={local.task} "
          f"{'(train_dataset='+lm_name+')' if lm_name else ''}")
    train_one_task(local)


def main():
    p = argparse.ArgumentParser()
    # Multi-selection
    p.add_argument(
        "--datasets", nargs="+",
        default=["c4-small", "triviaqa", "truthfulqa"],
        help="Datasets to fine-tune on (choices: c4-small, c4, triviaqa, truthfulqa)"
    )
    p.add_argument(
        "--variants", nargs="+",
        default=["baseline", "thermal"],
        choices=["baseline", "thermal"],
        help="Model variants to run."
    )

    # Core
    p.add_argument("--variant", type=str, default="thermal", choices=["baseline", "thermal"])  # single-run fallback
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--task", type=str, default="lm", choices=["lm","triviaqa","truthfulqa"], help="Internal use")
    p.add_argument("--train_dataset", type=str, default="c4-small", help="For task=lm: c4 or c4-small")

    # Train hparams
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--grad_accum_steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    # Thermal free-energy knobs
    p.add_argument("--alpha_entropy", type=float, default=0.0, help="α in −α·τ·H")
    p.add_argument("--lambda_reg", type=float, default=0.0, help="λ for logτ²")
    p.add_argument("--tau_clamp_min", type=float, default=-2.0)
    p.add_argument("--tau_clamp_max", type=float, default=1.5)
    p.add_argument("--tau_smooth", action="store_true")

    # Fairness / LoRA
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--train_lm_head", type=str, default="all", choices=["none","bias","all"])
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_lr", type=float, default=1e-4)

    # Saving / telemetry / eval cadence
    p.add_argument("--save_dir", type=str, default="checkpoints_llama")
    p.add_argument("--eval_every", type=int, default=20, help="steps between evals")
    p.add_argument("--log_every", type=int, default=20, help="steps between telemetry logs")
    p.add_argument("--patience_steps", type=int, default=400, help="early stop if no improvement for this many steps (0=off)")
    p.add_argument("--wandb", action="store_true")

    args = p.parse_args()

    # If the user only wants a single run via legacy flags, honor that:
    if args.datasets == [] and args.variants == []:
        train_one_task(args)
        return

    # Loop over all combinations
    for variant in args.variants:
        for dataset in args.datasets:
            run_experiment(args, dataset, variant)


if __name__ == "__main__":
    main()