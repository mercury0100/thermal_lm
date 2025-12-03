#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make core evaluation plots for Thermal LMs (GPT-2, multi-dataset).

Outputs (into --out_dir), each as a 3-panel figure (one panel per dataset):
  - fig_ppl_ece.png
  - fig_reliability_baseline.png
  - fig_reliability_thermal.png
  - fig_tau_entropy.png
  - fig_synth_ambiguity.png
  - fig_risk_coverage.png

Example:

python make_plots_gpt2.py \
  --ckpt_dir checkpoints_gpt2 \
  --out_dir figures \
  --datasets c4-small wikitext-103 nq_open \
  --max_batches 20
"""

import os
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from datasets import load_dataset, disable_progress_bar
from transformers import AutoTokenizer, AutoModelForCausalLM

from thermal_lm import ThermalLMForCausalLM

disable_progress_bar()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BAR", "1")


# ----------------------------
# LoRA helpers (match training)
# ----------------------------
def try_import_peft():
    try:
        from peft import LoraConfig, get_peft_model
        return LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "PEFT is required when loading LoRA checkpoints. Install with `pip install peft`."
        ) from e


def add_lora(model, rank: int, alpha: int, dropout: float):
    """
    Wrap model with LoRA adapters on GPT-2-style projection modules.
    Must match the configuration used during training.
    """
    LoraConfig, get_peft_model = try_import_peft()
    targets = ["c_attn", "c_fc", "c_proj"]  # GPT-2 block projections
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    return get_peft_model(model, cfg)


# ----------------------------
# Utils: loading checkpoints
# ----------------------------
def _pick_best_dir(base_dir: Path) -> Path:
    """
    For a given variant/dataset directory like:
        ckpt_dir / "thermal_wikitext-103_gpt2"
    pick the latest 'best_step*_ppl*' subdirectory.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {base_dir}")
    cands = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("best_step")],
        key=lambda p: p.name,
    )
    if not cands:
        raise FileNotFoundError(f"No best_step* subdirs in {base_dir}")
    return cands[-1]


def load_models_for_dataset(
    ckpt_root: str,
    dataset_tag: str,
    device: torch.device,
) -> Tuple[AutoTokenizer, torch.nn.Module, torch.nn.Module, Dict[str, str]]:
    """
    Load baseline + thermal models + tokenizer for a specific dataset.

    Expects training layout:
        ckpt_root / "baseline_<dataset>_gpt2" / best_step... / pytorch_model.bin
        ckpt_root / "thermal_<dataset>_gpt2"  / best_step... / pytorch_model.bin

    This reconstructs the SAME architecture used in training, including LoRA,
    so perplexities match the training-time validation numbers.
    """
    ckpt_root = Path(ckpt_root)

    base_dir = ckpt_root / f"baseline_{dataset_tag}_gpt2"
    th_dir = ckpt_root / f"thermal_{dataset_tag}_gpt2"

    base_best = _pick_best_dir(base_dir)
    th_best = _pick_best_dir(th_dir)

    # Load train args to recover model + LoRA + thermal settings
    with open(base_best / "train_args.json", "r") as f:
        base_args = json.load(f)
    with open(th_best / "train_args.json", "r") as f:
        th_args = json.load(f)

    base_model_name = base_args.get("base_model", "gpt2")

    # Tokenizer from baseline best dir (has any added tokens)
    tokenizer = AutoTokenizer.from_pretrained(base_best.as_posix())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ----- Baseline model -----
    m_base = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    if base_args.get("use_lora", False):
        m_base = add_lora(
            m_base,
            rank=base_args.get("lora_rank", 8),
            alpha=base_args.get("lora_alpha", 16),
            dropout=base_args.get("lora_dropout", 0.1),
        )
    sd_base = torch.load(base_best / "pytorch_model.bin", map_location=device)
    m_base.load_state_dict(sd_base, strict=True)
    m_base.eval()

    # ----- Thermal model -----
    m_th = ThermalLMForCausalLM.from_base_pretrained(
        base_model_name_or_path=base_model_name,
        tau_clamp_min=th_args.get("tau_clamp_min", -2.0),
        tau_clamp_max=th_args.get("tau_clamp_max", 1.5),
        lambda_reg=th_args.get("lambda_reg", 0.0),
        alpha_entropy=th_args.get("alpha_entropy", 0.0),
        tau_smooth=th_args.get("tau_smooth", False),
    ).to(device)

    if th_args.get("use_lora", False):
        m_th = add_lora(
            m_th,
            rank=th_args.get("lora_rank", 8),
            alpha=th_args.get("lora_alpha", 16),
            dropout=th_args.get("lora_dropout", 0.1),
        )

    sd_th = torch.load(th_best / "pytorch_model.bin", map_location=device)
    m_th.load_state_dict(sd_th, strict=True)
    m_th.eval()

    paths = {
        "baseline": base_best.as_posix(),
        "thermal": th_best.as_posix(),
    }
    return tokenizer, m_base, m_th, paths


# ----------------------------
# Data: val loaders per dataset
# ----------------------------
def build_val_loader_c4(tokenizer, name: str, block_size: int, batch_size: int) -> DataLoader:
    """C4 / C4-small (streaming) → fixed-length LM blocks without padding."""
    from itertools import islice
    name = name.lower()
    assert name in ["c4", "c4-small"], f"Unsupported dataset for LM: {name}"
    if name == "c4":
        _, val_blocks = 50000, 2000
    else:
        _, val_blocks = 10000, 1000

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

    va_it = ds["validation"].shuffle(seed=43)
    val_ids = list(islice(stream_blocks(va_it, val_blocks), val_blocks))

    val = TensorDataset(
        torch.stack(val_ids),
        torch.ones_like(val_ids[0]).repeat(len(val_ids), 1),
    )

    def collate(b):
        x = torch.stack([t[0] for t in b])
        m = torch.stack([t[1] for t in b])
        return {"input_ids": x, "attention_mask": m}

    return DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate)


def build_val_loader_wikitext103(tokenizer, block_size: int, batch_size: int) -> DataLoader:
    """wikitext-103-raw-v1 → fixed-length LM blocks."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    val_ds = ds["validation"]

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

    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"]).map(
        group_texts, batched=True
    )
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False)


def build_val_loader_nq_open(
    tokenizer,
    block_size: int,
    batch_size: int,
    max_train_examples: Optional[int] = None,
    val_frac: float = 0.05,
) -> DataLoader:
    """
    nq_open → LM over 'Q: ...\\nA: ...' strings.
    For evaluation we just use a held-out split from the train set (same recipe as training).
    """
    ds = load_dataset("google-research-datasets/nq_open")
    raw = ds["train"]

    if max_train_examples is not None and max_train_examples > 0 and len(raw) > max_train_examples:
        raw = raw.shuffle(seed=123).select(range(max_train_examples))

    split = raw.train_test_split(test_size=val_frac, seed=321)
    val_raw = split["test"]

    def qa_to_text(ex):
        q = ex.get("question", "")
        ans = ex.get("answer", [])
        if isinstance(ans, list) and len(ans) > 0:
            a = ans[0]
        else:
            a = str(ans) if ans is not None else ""
        text = f"Q: {q}\nA: {a}"
        return {"text": text}

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

    val_ds = val_raw.map(tokenize_fn, batched=True, remove_columns=val_raw.column_names).map(
        group_texts, batched=True
    )
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False)


def build_val_loader_for_dataset(
    dataset_tag: str,
    tokenizer,
    block_size: int,
    batch_size: int,
) -> DataLoader:
    key = dataset_tag.lower()
    if key in {"c4-small", "c4"}:
        return build_val_loader_c4(tokenizer, "c4-small" if key == "c4-small" else "c4",
                                   block_size, batch_size)
    if key in {"wikitext-103", "wikitext103"}:
        return build_val_loader_wikitext103(tokenizer, block_size, batch_size)
    if key in {"nq_open", "natural-questions", "naturalquestions"}:
        return build_val_loader_nq_open(tokenizer, block_size, batch_size)
    raise ValueError(f"Unknown dataset token for val loader: {dataset_tag}")


# ----------------------------
# Metrics: PPL, ECE, Reliability
# ----------------------------
@torch.no_grad()
def ece_from_logits(logits, labels, n_bins=20):
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()

    conf = conf.flatten().cpu().numpy()
    correct = correct.flatten().cpu().numpy()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    diag = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = (conf > lo) & (conf <= hi)
        if idx.sum() == 0:
            diag.append((0.0, 0.0, 0))
            continue
        acc = correct[idx].mean()
        c = conf[idx].mean()
        w = idx.mean()
        ece += w * abs(acc - c)
        diag.append((c, acc, idx.sum()))
    return float(ece), diag


# ----------------------------
# Low-level helpers
# ----------------------------
@torch.no_grad()
def untempered_logits_gpt2(model, tokens, attn):
    """For GPT-2 Thermal wrapper: get logits without τ scaling (intrinsic branching)."""
    if hasattr(model, "transformer"):
        h = model.transformer(input_ids=tokens, attention_mask=attn).last_hidden_state
        logits_base = model.lm_head(h)
    else:
        logits_base = model(input_ids=tokens, attention_mask=attn).logits
    return logits_base


@torch.no_grad()
def eval_epoch(model, loader, device, variant="baseline", max_batches=None, for_tau_entropy=False):
    model.eval()
    total_nll, total_tokens = 0.0, 0
    all_logits_for_ece = []
    all_labels_for_ece = []
    tau_list = []
    ent_list = []
    acc_list = []

    seen = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        tokens = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        attn = attn[:, :-1]

        if variant == "thermal":
            out = model(input_ids=tokens, attention_mask=attn)
            logits_t = out.logits  # tempered logits
            B, T, V = logits_t.size()
            nll = F.cross_entropy(logits_t.reshape(-1, V), labels.reshape(-1), reduction="sum")
            total_nll += nll.item()
            total_tokens += B * T

            all_logits_for_ece.append(logits_t.detach().cpu())
            all_labels_for_ece.append(labels.detach().cpu())

            if for_tau_entropy:
                tau = out.aux["tau"].detach()  # (B,T)
                probs = torch.softmax(logits_t, dim=-1)
                ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                pred = probs.argmax(dim=-1)
                acc = (pred == labels).float()

                tau_list.append(tau.flatten().cpu().numpy())
                ent_list.append(ent.flatten().cpu().numpy())
                acc_list.append(acc.flatten().cpu().numpy())

        else:
            logits = model(input_ids=tokens, attention_mask=attn).logits
            B, T, V = logits.size()
            nll = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="sum")
            total_nll += nll.item()
            total_tokens += B * T
            all_logits_for_ece.append(logits.detach().cpu())
            all_labels_for_ece.append(labels.detach().cpu())

            if for_tau_entropy:
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                pred = probs.argmax(dim=-1)
                acc = (pred == labels).float()
                tau = torch.zeros_like(ent)
                tau_list.append(tau.flatten().cpu().numpy())
                ent_list.append(ent.flatten().cpu().numpy())
                acc_list.append(acc.flatten().cpu().numpy())

        seen += 1
        if max_batches and seen >= max_batches:
            break

    ppl = math.exp(total_nll / max(1, total_tokens))
    all_logits_for_ece = torch.cat(all_logits_for_ece, dim=0)
    all_labels_for_ece = torch.cat(all_labels_for_ece, dim=0)

    out = {
        "ppl": ppl,
        "logits_for_ece": all_logits_for_ece,
        "labels_for_ece": all_labels_for_ece,
    }
    if for_tau_entropy:
        out["tau_vec"] = np.concatenate(tau_list, axis=0)
        out["ent_vec"] = np.concatenate(ent_list, axis=0)
        out["acc_vec"] = np.concatenate(acc_list, axis=0)
    return out


# ----------------------------
# Plot helpers (multi-dataset)
# ----------------------------
def plot_ppl_ece_multi(results_by_ds: Dict[str, Dict], out_dir: str):
    out_dir = Path(out_dir)
    datasets = list(results_by_ds.keys())
    n = len(datasets)

    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        res = results_by_ds[ds]
        base_eval = res["baseline_eval"]
        therm_eval = res["thermal_eval"]

        ece_b, _ = ece_from_logits(base_eval["logits_for_ece"], base_eval["labels_for_ece"])
        ece_t, _ = ece_from_logits(therm_eval["logits_for_ece"], therm_eval["labels_for_ece"])

        labels = ["Baseline", "Thermal"]
        ppl = [base_eval["ppl"], therm_eval["ppl"]]
        ece = [ece_b, ece_t]
        x = np.arange(2)
        w = 0.35
        ax1 = ax
        ax2 = ax1.twinx()
        ax1.bar(x - w / 2, ppl, width=w, label="PPL")
        ax2.bar(x + w / 2, ece, width=w, label="ECE", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=0)
        ax1.set_title(ds)
        if ax is axes[0]:
            ax1.set_ylabel("Perplexity")
        ax2.set_ylabel("ECE")

    fig.suptitle("Perplexity & ECE (validation)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_dir / "fig_ppl_ece.png", dpi=250)
    plt.close(fig)

    # Reliability diagrams: baseline & thermal separately
    # baseline
    fig_b, axes_b = plt.subplots(1, n, figsize=(4 * n, 4), sharex=True, sharey=True)
    if n == 1:
        axes_b = [axes_b]
    for ax, ds in zip(axes_b, datasets):
        base_eval = results_by_ds[ds]["baseline_eval"]
        _, diag_b = ece_from_logits(base_eval["logits_for_ece"], base_eval["labels_for_ece"])
        xs = [c for (c, a, cnt) in diag_b if cnt > 0]
        ys = [a for (c, a, cnt) in diag_b if cnt > 0]
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.scatter(xs, ys, s=10)
        ax.set_title(ds)
        if ax is axes_b[0]:
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
        else:
            ax.set_xlabel("Confidence")
    fig_b.suptitle("Reliability: Baseline")
    fig_b.tight_layout()
    fig_b.subplots_adjust(top=0.88)
    fig_b.savefig(out_dir / "fig_reliability_baseline.png", dpi=250)
    plt.close(fig_b)

    # thermal
    fig_t, axes_t = plt.subplots(1, n, figsize=(4 * n, 4), sharex=True, sharey=True)
    if n == 1:
        axes_t = [axes_t]
    for ax, ds in zip(axes_t, datasets):
        therm_eval = results_by_ds[ds]["thermal_eval"]
        _, diag_t = ece_from_logits(therm_eval["logits_for_ece"], therm_eval["labels_for_ece"])
        xs = [c for (c, a, cnt) in diag_t if cnt > 0]
        ys = [a for (c, a, cnt) in diag_t if cnt > 0]
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.scatter(xs, ys, s=10)
        ax.set_title(ds)
        if ax is axes_t[0]:
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
        else:
            ax.set_xlabel("Confidence")
    fig_t.suptitle("Reliability: Thermal")
    fig_t.tight_layout()
    fig_t.subplots_adjust(top=0.88)
    fig_t.savefig(out_dir / "fig_reliability_thermal.png", dpi=250)
    plt.close(fig_t)


def plot_tau_entropy_multi(results_by_ds: Dict[str, Dict], out_dir: str, n_quant: int = 10):
    out_dir = Path(out_dir)
    datasets = list(results_by_ds.keys())
    n = len(datasets)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        therm_eval = results_by_ds[ds]["thermal_eval_tau"]
        tau = therm_eval["tau_vec"]
        ent = therm_eval["ent_vec"]
        acc = therm_eval["acc_vec"]

        qs = np.quantile(tau, np.linspace(0, 1, n_quant + 1))
        xs = []
        mean_ent = []
        mean_acc = []
        for i in range(n_quant):
            lo, hi = qs[i], qs[i + 1]
            idx = (tau >= lo) & (tau <= hi if i == n_quant - 1 else tau < hi)
            if idx.sum() == 0:
                xs.append((lo + hi) / 2)
                mean_ent.append(np.nan)
                mean_acc.append(np.nan)
                continue
            xs.append((lo + hi) / 2)
            mean_ent.append(ent[idx].mean())
            mean_acc.append(acc[idx].mean())

        r = np.corrcoef(tau, ent)[0, 1] if np.isfinite(ent).all() else np.nan

        ax2 = ax.twinx()
        ax.plot(xs, mean_ent, marker="o", label="Entropy")
        ax2.plot(xs, mean_acc, marker="s", linestyle="--", label="Accuracy", color="tab:orange")
        
        # Add combined legend
        lines_ent, labels_ent = ax.get_legend_handles_labels()
        lines_acc, labels_acc = ax2.get_legend_handles_labels()
        ax.legend(lines_ent + lines_acc, labels_ent + labels_acc, loc="best")
        
        ax.set_title(f"{ds} (r={r:.2f})")
        ax.set_xlabel("τ (quantile centers)")
        if ax is axes[0]:
            ax.set_ylabel("Predictive entropy")
            ax2.set_ylabel("Top-1 accuracy")

    fig.suptitle("τ–Entropy alignment")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_dir / "fig_tau_entropy.png", dpi=250)
    plt.close(fig)


# ----------------------------
# Synthetic ambiguity probe (per dataset)
# ----------------------------
@torch.no_grad()
def tau_for_first_generated_token(model_th, tokenizer, prompt, device, top_p=0.95, temperature=1.0):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)

    out0 = model_th(input_ids=input_ids, attention_mask=attn)
    logits = out0.logits[:, -1, :] / max(1e-8, temperature)
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        k = int(torch.searchsorted(cum[0], torch.tensor(top_p, device=cum.device)).item()) + 1
        k = max(1, min(k, sorted_probs.size(-1)))
        keep_idx = sorted_idx[:, :k]
        keep_probs = sorted_probs[:, :k] / sorted_probs[:, :k].sum(dim=-1, keepdim=True)
        next_id = keep_idx[0, torch.multinomial(keep_probs[0], num_samples=1)]
    else:
        next_id = torch.argmax(probs[0], dim=-1)

    input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
    attn = torch.ones_like(input_ids, device=device)
    out1 = model_th(input_ids=input_ids, attention_mask=attn)
    tau_slot_after = float(out1.aux["tau"][0, -1].item())
    return tau_slot_after


@torch.no_grad()
def collect_tau_on_prompts(model_th, tokenizer, prompts, device, top_p=0.95, temperature=1.0):
    vals = []
    for p in prompts:
        vals.append(tau_for_first_generated_token(model_th, tokenizer, p, device, top_p, temperature))
    return np.array(vals, dtype=np.float32)


def plot_synth_ambiguity_multi(synth_by_ds: Dict[str, Dict[str, np.ndarray]], out_dir: str):
    out_dir = Path(out_dir)
    datasets = list(synth_by_ds.keys())
    n = len(datasets)

    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        tau_det = synth_by_ds[ds]["tau_det"]
        tau_open = synth_by_ds[ds]["tau_open"]
        data = [tau_det, tau_open]
        ax.boxplot(data, labels=["Deterministic", "Open-class"], showfliers=False)
        ax.set_title(ds)
        if ax is axes[0]:
            ax.set_ylabel("τ at slot")

    fig.suptitle("Synthetic ambiguity probe (τ higher on open-class?)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_dir / "fig_synth_ambiguity.png", dpi=250)
    plt.close(fig)


# ----------------------------
# Risk–coverage (τ vs confidence)
# ----------------------------
@torch.no_grad()
def risk_coverage_curves(model_b, model_th, loader, device, max_batches=None):
    logits_b_list = []
    logits_t_base_list = []
    labels_list = []
    tau_list = []

    seen = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        tokens = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        attn = attn[:, :-1]

        logits_b = model_b(input_ids=tokens, attention_mask=attn).logits
        logits_b_list.append(logits_b.detach().cpu())

        out_th = model_th(input_ids=tokens, attention_mask=attn)
        tau = out_th.aux["tau"].detach().cpu()
        tau_list.append(tau)
        logits_t_base = untempered_logits_gpt2(model_th, tokens, attn).detach().cpu()
        logits_t_base_list.append(logits_t_base)

        labels_list.append(labels.detach().cpu())

        seen += 1
        if max_batches and seen >= max_batches:
            break

    logits_b = torch.cat(logits_b_list, dim=0)
    logits_t_base = torch.cat(logits_t_base_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    tau = torch.cat(tau_list, dim=0)

    probs_b = torch.softmax(logits_b, dim=-1)
    conf_b, pred_b = probs_b.max(dim=-1)
    correct_b = (pred_b == labels).float()

    probs_t_base = torch.softmax(logits_t_base, dim=-1)
    pred_t = probs_t_base.argmax(dim=-1)
    correct_t = (pred_t == labels).float()

    cov_grid = np.linspace(1.0, 0.2, 9)
    conf_vals = conf_b.flatten().numpy()
    tau_vals = tau.flatten().numpy()

    res_baseline = []
    res_thermal = []

    for cov in cov_grid:
        # confidence-based coverage
        t_conf = np.quantile(conf_vals, 1.0 - cov)
        keep_b = (conf_b >= t_conf).float()
        acc_b = (correct_b * keep_b).sum().item() / max(1.0, keep_b.sum().item())
        res_baseline.append((cov, acc_b))

        # τ-based coverage (abstain on high τ)
        t_tau = np.quantile(tau_vals, cov)  # keep τ < t_tau
        keep_t = (tau < t_tau).float()
        acc_t = (correct_t * keep_t).sum().item() / max(1.0, keep_t.sum().item())
        res_thermal.append((cov, acc_t))

    return res_baseline, res_thermal


def plot_risk_coverage_multi(risk_by_ds: Dict[str, Dict[str, List[Tuple[float, float]]]], out_dir: str):
    out_dir = Path(out_dir)
    datasets = list(risk_by_ds.keys())
    n = len(datasets)

    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        res_b = risk_by_ds[ds]["baseline"]
        res_t = risk_by_ds[ds]["thermal"]
        cov_b, acc_b = zip(*res_b)
        cov_t, acc_t = zip(*res_t)
        ax.plot(cov_b, acc_b, marker="o", label="Confidence (Baseline)")
        ax.plot(cov_t, acc_t, marker="s", label="τ (Thermal)")
        ax.set_title(ds)
        ax.set_xlabel("Coverage")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy (kept tokens)")
        ax.legend()

    fig.suptitle("Risk–coverage (validation)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_dir / "fig_risk_coverage.png", dpi=250)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints_gpt2")
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["c4-small", "wikitext-103", "nq_open"],
        help="Datasets: e.g. c4-small wikitext-103 nq_open",
    )
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=200, help="cap eval batches for speed")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accumulate results per dataset
    results_by_ds: Dict[str, Dict] = {}
    synth_by_ds: Dict[str, Dict[str, np.ndarray]] = {}
    risk_by_ds: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    for ds_tag in args.datasets:
        print(f"\n=== Dataset: {ds_tag} ===")
        tokenizer, model_b, model_th, paths = load_models_for_dataset(args.ckpt_dir, ds_tag, device)
        print(f"[Loaded] baseline: {paths['baseline']}")
        print(f"[Loaded] thermal : {paths['thermal']}")

        val_loader = build_val_loader_for_dataset(
            ds_tag, tokenizer, block_size=args.block_size, batch_size=args.batch_size
        )

        # Eval (PPL/ECE, τ–entropy)
        base_eval = eval_epoch(
            model_b,
            val_loader,
            device,
            variant="baseline",
            max_batches=args.max_batches,
            for_tau_entropy=False,
        )
        therm_eval = eval_epoch(
            model_th,
            val_loader,
            device,
            variant="thermal",
            max_batches=args.max_batches,
            for_tau_entropy=True,
        )
        print(f"[PPL-{ds_tag}] baseline={base_eval['ppl']:.2f} | thermal={therm_eval['ppl']:.2f}")

        # Copy tau eval separately for clarity
        therm_eval_tau = {
            "tau_vec": therm_eval["tau_vec"],
            "ent_vec": therm_eval["ent_vec"],
            "acc_vec": therm_eval["acc_vec"],
        }

        results_by_ds[ds_tag] = {
            "baseline_eval": base_eval,
            "thermal_eval": therm_eval,
            "thermal_eval_tau": therm_eval_tau,
            "paths": paths,
        }

        # Synthetic ambiguity probe for this dataset's thermal model
        det_prompts = [
            "2 + 2 = ",
            "Paris is the capital of ",
            "JPY is the national currency of ",
            "H2O is the chemical symbol for ",
            "Einstein's first name is ",
            "The opposite of hot is ",
            "A week has seven ",
            "January is the first month of the ",
        ] * 25

        open_prompts = [
            "He lived in France. ",
            "To conclude, ",
            "I must say, ",
            "The story ended with a",
            "She bought a ",
            "The discovery suggests that ",
            "The time was ripe. ",
            "Their favorite movie is Jurassic Park, however, the ",
        ] * 25

        tau_det = collect_tau_on_prompts(model_th, tokenizer, det_prompts, device=device)
        tau_open = collect_tau_on_prompts(model_th, tokenizer, open_prompts, device=device)

        synth_by_ds[ds_tag] = {
            "tau_det": tau_det,
            "tau_open": tau_open,
        }

        # Risk–coverage curves for this dataset
        res_b, res_t = risk_coverage_curves(
            model_b, model_th, val_loader, device, max_batches=args.max_batches
        )
        risk_by_ds[ds_tag] = {
            "baseline": res_b,
            "thermal": res_t,
        }

    # 1) PPL & ECE (+ reliability)
    plot_ppl_ece_multi(results_by_ds, args.out_dir)

    # 2) τ–entropy alignment
    plot_tau_entropy_multi(results_by_ds, args.out_dir, n_quant=10)

    # 3) Synthetic ambiguity probe
    plot_synth_ambiguity_multi(synth_by_ds, args.out_dir)

    # 4) Risk–coverage curves
    plot_risk_coverage_multi(risk_by_ds, args.out_dir)

    # Save a tiny report
    report = {}
    for ds in args.datasets:
        report[ds] = {
            "baseline_ckpt": results_by_ds[ds]["paths"]["baseline"],
            "thermal_ckpt": results_by_ds[ds]["paths"]["thermal"],
            "ppl": {
                "baseline": results_by_ds[ds]["baseline_eval"]["ppl"],
                "thermal": results_by_ds[ds]["thermal_eval"]["ppl"],
            },
        }
    with open(Path(args.out_dir) / "summary_multi.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Done] Figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()