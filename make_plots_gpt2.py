#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-ready diagnostics for Thermal LMs (calibrated σ² head).

For each dataset, this script:
  - loads baseline and thermal checkpoints,
  - computes PPL (baseline vs thermal),
  - collects token-level CE and σ² for the thermal model,
  - plots CE–σ² alignment (binned curve + corr),
  - plots risk–coverage curves (baseline confidence vs σ²-based abstention),
  - runs a synthetic ambiguity probe (deterministic vs open prompts) on σ².

Outputs (into --out_dir):
  - fig_ce_sigma2.png
  - fig_risk_coverage.png
  - fig_synth_ambiguity.png
  - summary_multi.json

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
            "PEFT is required when loading LoRA checkpoints. "
            "Install with `pip install peft`."
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
):
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

    # ----- Thermal model (calibrated σ² head) -----
    m_th = ThermalLMForCausalLM.from_base_pretrained(
        base_model_name_or_path=base_model_name,
        logsigma_clamp_min=th_args.get("tau_clamp_min", -6.0),
        logsigma_clamp_max=th_args.get("tau_clamp_max", 2.0),
        lambda_reg=th_args.get("lambda_reg", 0.0),
        sigma_smooth=th_args.get("tau_smooth", False),
        noise_at_eval=False,
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
    return tokenizer, m_base, m_th, paths, base_args, th_args


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
# Core evaluation: CE, σ², PPL, etc.
# ----------------------------
@torch.no_grad()
def collect_stats(
    model_b,
    model_th,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
):
    """
    Collect token-level statistics from baseline and thermal models:

      - baseline_ppl, thermal_ppl
      - arrays of CE_t (thermal), σ_t^2, entropy_t (thermal)
      - baseline confidence & correctness
      - thermal correctness (for risk–coverage)
    """
    model_b.eval()
    model_th.eval()

    total_nll_base = 0.0
    total_nll_th = 0.0
    total_tokens = 0

    ce_list = []
    sigma2_list = []
    ent_list = []

    conf_b_list = []
    corr_b_list = []
    corr_th_list = []

    seen = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        tokens = ids[:, :-1]
        labels = ids[:, 1:]
        attn_tok = attn[:, :-1]
        valid = attn_tok > 0  # (B,T)

        # ----- Baseline -----
        out_b = model_b(input_ids=tokens, attention_mask=attn_tok)
        logits_b = out_b.logits                          # (B,T,V)
        B, T, V = logits_b.size()

        logits_b_flat = logits_b.reshape(-1, V)
        labels_flat = labels.reshape(-1).clamp_min(0)
        ce_b_flat = F.cross_entropy(
            logits_b_flat, labels_flat, reduction="none"
        ).reshape(B, T)


        ce_b_valid = ce_b_flat[valid]
        total_nll_base += float(ce_b_valid.sum().item())

        probs_b = torch.softmax(logits_b, dim=-1)
        conf_b, pred_b = probs_b.max(dim=-1)
        correct_b = (pred_b == labels).float()

        conf_b_list.append(conf_b[valid].cpu().numpy())
        corr_b_list.append(correct_b[valid].cpu().numpy())

        # ----- Thermal (same logits distribution, plus σ) -----
        out_th = model_th(input_ids=tokens, attention_mask=attn_tok)
        logits_th = out_th.logits                        # (B,T,V)
        sigma = out_th.aux["sigma"]              # (B,T) aligned with tokens
        sigma2 = sigma ** 2

        logits_th_flat = logits_th.reshape(-1, V)
        ce_th_flat = F.cross_entropy(
            logits_th_flat, labels_flat, reduction="none"
        ).reshape(B, T)

        ce_th_valid = ce_th_flat[valid]
        total_nll_th += float(ce_th_valid.sum().item())

        probs_th = torch.softmax(logits_th, dim=-1)
        ent = -(probs_th * probs_th.clamp_min(1e-12).log()).sum(dim=-1)
        pred_th = probs_th.argmax(dim=-1)
        correct_th = (pred_th == labels).float()

        ce_list.append(ce_th_valid.cpu().numpy())
        sigma2_list.append(sigma2[valid].cpu().numpy())
        ent_list.append(ent[valid].cpu().numpy())
        corr_th_list.append(correct_th[valid].cpu().numpy())

        total_tokens += int(valid.sum().item())

        seen += 1
        if max_batches is not None and seen >= max_batches:
            break

    baseline_ppl = math.exp(total_nll_base / max(1, total_tokens))
    thermal_ppl = math.exp(total_nll_th / max(1, total_tokens))

    ce_all = np.concatenate(ce_list, axis=0)
    sigma2_all = np.concatenate(sigma2_list, axis=0)
    ent_all = np.concatenate(ent_list, axis=0)
    conf_b_all = np.concatenate(conf_b_list, axis=0)
    corr_b_all = np.concatenate(corr_b_list, axis=0)
    corr_th_all = np.concatenate(corr_th_list, axis=0)

    return {
        "baseline_ppl": baseline_ppl,
        "thermal_ppl": thermal_ppl,
        "ce": ce_all,
        "sigma2": sigma2_all,
        "entropy": ent_all,
        "conf_b": conf_b_all,
        "corr_b": corr_b_all,
        "corr_th": corr_th_all,
    }


# ----------------------------
# Synthetic ambiguity probe
# ----------------------------
@torch.no_grad()
def sigma2_for_first_generated_token(
    model_th,
    tokenizer,
    prompt: str,
    device: torch.device,
    top_p: float = 0.95,
    temperature: float = 1.0,
) -> float:
    """
    For a given prompt, generate a single next token, then return σ^2 at that new position.
    """
    model_th.eval()

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)

    # logits at last position of prompt
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

    # Append sampled token and recompute σ^2 at that position
    input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
    attn = torch.ones_like(input_ids, device=device)
    out1 = model_th(input_ids=input_ids, attention_mask=attn)
    sigma = out1.aux["sigma"][0, -1].item()
    return float(sigma ** 2)


@torch.no_grad()
def collect_sigma2_on_prompts(
    model_th,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    top_p: float = 0.95,
    temperature: float = 1.0,
) -> np.ndarray:
    vals = []
    for p in prompts:
        vals.append(sigma2_for_first_generated_token(model_th, tokenizer, p, device, top_p, temperature))
    return np.array(vals, dtype=np.float32)


# ----------------------------
# Risk–coverage curves (σ² vs confidence)
# ----------------------------
def compute_risk_coverage(
    conf_b: np.ndarray,
    corr_b: np.ndarray,
    sigma2: np.ndarray,
    corr_th: np.ndarray,
    cov_grid: Optional[np.ndarray] = None,
):
    """
    Compute risk–coverage curves:
      - baseline: coverage by descending confidence
      - thermal: coverage by ascending σ² (abstain on high σ²)
    """
    if cov_grid is None:
        cov_grid = np.linspace(1.0, 0.2, 9)

    conf_vals = conf_b
    sigma2_vals = sigma2

    res_baseline = []
    res_thermal = []

    for cov in cov_grid:
        # Confidence-based coverage
        t_conf = np.quantile(conf_vals, 1.0 - cov)
        keep_b = conf_vals >= t_conf
        if keep_b.sum() > 0:
            acc_b = corr_b[keep_b].mean()
        else:
            acc_b = np.nan
        res_baseline.append((cov, acc_b))

        # σ²-based coverage (abstain on high σ²)
        t_sigma = np.quantile(sigma2_vals, cov)  # keep σ² < t_sigma
        keep_t = sigma2_vals < t_sigma
        if keep_t.sum() > 0:
            acc_t = corr_th[keep_t].mean()
        else:
            acc_t = np.nan
        res_thermal.append((cov, acc_t))

    return res_baseline, res_thermal


# ----------------------------
# Plot helpers
# ----------------------------
def plot_ce_sigma2_multi(results_by_ds: Dict[str, Dict], out_dir: str, n_bins: int = 20):
    """
    For each dataset: plot binned σ² vs CE, with corr(CE,σ²) in the title.
    """
    out_dir = Path(out_dir)
    datasets = list(results_by_ds.keys())
    n = len(datasets)

    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.8), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        stats = results_by_ds[ds]["stats"]
        ce = stats["ce"]
        sigma2 = stats["sigma2"]

        # correlation
        r = float(np.corrcoef(ce, sigma2)[0, 1])

        # bin CE and compute mean σ² per bin
        qs = np.quantile(ce, np.linspace(0, 1, n_bins + 1))
        ce_centers = []
        sigma_means = []
        for i in range(n_bins):
            lo, hi = qs[i], qs[i + 1]
            if i < n_bins - 1:
                mask = (ce >= lo) & (ce < hi)
            else:
                mask = (ce >= lo) & (ce <= hi)
            if mask.sum() == 0:
                continue
            ce_centers.append(0.5 * (lo + hi))
            sigma_means.append(sigma2[mask].mean())

        ax.plot(ce_centers, sigma_means, marker="o", linewidth=1.8)
        ax.set_title(f"{ds} (r={r:.2f})")
        ax.set_xlabel("Token NLL (cross-entropy)")
        if ax is axes[0]:
            ax.set_ylabel(r"Mean $\sigma_t^2$")

    fig.suptitle(r"Alignment between token NLL and microscopic temperature $\sigma_t^2$")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_dir / "fig_ce_sigma2.png", dpi=250)
    plt.close(fig)


def plot_risk_coverage_multi(risk_by_ds: Dict[str, Dict[str, List[Tuple[float, float]]]], out_dir: str):
    """
    Plot risk–coverage curves for each dataset.
    """
    out_dir = Path(out_dir)
    datasets = list(risk_by_ds.keys())
    n = len(datasets)

    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        res_b = risk_by_ds[ds]["baseline"]
        res_t = risk_by_ds[ds]["thermal"]
        cov_b, acc_b = zip(*res_b)
        cov_t, acc_t = zip(*res_t)
        ax.plot(cov_b, acc_b, marker="o", label="Confidence (baseline)")
        ax.plot(cov_t, acc_t, marker="s", label=r"$\sigma^2$ (thermal)")
        ax.set_title(ds)
        ax.set_xlabel("Coverage")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy on kept tokens")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left")

    fig.suptitle("Risk–coverage: confidence vs microscopic temperature gating")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_dir / "fig_risk_coverage.png", dpi=250)
    plt.close(fig)


def plot_synth_ambiguity_multi(synth_by_ds: Dict[str, Dict[str, np.ndarray]], out_dir: str):
    """
    Boxplots of σ² on deterministic vs open-class prompts.
    """
    out_dir = Path(out_dir)
    datasets = list(synth_by_ds.keys())
    n = len(datasets)

    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sigma2_det = synth_by_ds[ds]["sigma2_det"]
        sigma2_open = synth_by_ds[ds]["sigma2_open"]
        data = [sigma2_det, sigma2_open]
        ax.boxplot(data, labels=["Deterministic", "Open-class"], showfliers=False)
        ax.set_title(ds)
        if ax is axes[0]:
            ax.set_ylabel(r"$\sigma_t^2$ at generated slot")

    fig.suptitle(r"Synthetic ambiguity probe: $\sigma_t^2$ on deterministic vs open-class prompts")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_dir / "fig_synth_ambiguity.png", dpi=250)
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

    results_by_ds: Dict[str, Dict] = {}
    synth_by_ds: Dict[str, Dict[str, np.ndarray]] = {}
    risk_by_ds: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    summary = {}

    # Fixed prompts used across datasets
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

    for ds_tag in args.datasets:
        print(f"\n=== Dataset: {ds_tag} ===")
        tokenizer, model_b, model_th, paths, base_args, th_args = load_models_for_dataset(
            args.ckpt_dir, ds_tag, device
        )
        print(f"[Loaded] baseline: {paths['baseline']}")
        print(f"[Loaded] thermal : {paths['thermal']}")

        val_loader = build_val_loader_for_dataset(
            ds_tag, tokenizer, block_size=args.block_size, batch_size=args.batch_size
        )

        # Core token-level stats
        stats = collect_stats(
            model_b,
            model_th,
            val_loader,
            device,
            max_batches=args.max_batches,
        )

        print(
            f"[PPL-{ds_tag}] baseline={stats['baseline_ppl']:.2f} | "
            f"thermal={stats['thermal_ppl']:.2f}"
        )
        ce = stats["ce"]
        sigma2 = stats["sigma2"]
        corr_ce_sigma2 = float(np.corrcoef(ce, sigma2)[0, 1])
        print(f"[Corr-{ds_tag}] corr(CE, σ²) = {corr_ce_sigma2:.3f}")

        # Risk–coverage curves
        res_b, res_t = compute_risk_coverage(
            conf_b=stats["conf_b"],
            corr_b=stats["corr_b"],
            sigma2=stats["sigma2"],
            corr_th=stats["corr_th"],
        )
        risk_by_ds[ds_tag] = {
            "baseline": res_b,
            "thermal": res_t,
        }

        # Synthetic ambiguity probe for this dataset's thermal model
        sigma2_det = collect_sigma2_on_prompts(model_th, tokenizer, det_prompts, device=device)
        sigma2_open = collect_sigma2_on_prompts(model_th, tokenizer, open_prompts, device=device)

        synth_by_ds[ds_tag] = {
            "sigma2_det": sigma2_det,
            "sigma2_open": sigma2_open,
        }

        results_by_ds[ds_tag] = {
            "stats": stats,
            "paths": paths,
            "corr_ce_sigma2": corr_ce_sigma2,
        }

        summary[ds_tag] = {
            "baseline_ckpt": paths["baseline"],
            "thermal_ckpt": paths["thermal"],
            "ppl": {
                "baseline": stats["baseline_ppl"],
                "thermal": stats["thermal_ppl"],
            },
            "corr_ce_sigma2": corr_ce_sigma2,
        }

    # 1) CE–σ² alignment
    plot_ce_sigma2_multi(results_by_ds, args.out_dir, n_bins=20)

    # 2) Risk–coverage curves
    plot_risk_coverage_multi(risk_by_ds, args.out_dir)

    # 3) Synthetic ambiguity probe
    plot_synth_ambiguity_multi(synth_by_ds, args.out_dir)

    # Save a tiny report
    with open(Path(args.out_dir) / "summary_multi.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Done] Figures and summary saved to {args.out_dir}")


if __name__ == "__main__":
    main()
