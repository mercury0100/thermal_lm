#!/usr/bin/env python3
"""
sanity_check_span_ambiguity.py

Quick sanity-check script for span-level ambiguity in a HET-XL-style GPT-2 LM.

Idea:
- For each prompt, compute the final prompt hidden state.
- Sample k latent ambiguity vectors from the heteroscedastic heads.
- Reuse ONE sampled latent vector for the whole rollout span.
- Generate t new tokens per chain.
- Compare how much chains differ for ambiguous vs clear prompts.

This is meant as a smoke test, not a final benchmark.

Example:
    python test_gpt2.py \
        --model_dir ./gpt2-hetxl-kl \
        --train_script_path ./train_gpt2.py \
        --k 8 \
        --t 4 \
        --temperature 0. \
        --top_p 1.
"""

import argparse
import importlib.util
import math
import os
from itertools import combinations
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Config, set_seed


# -------------------------------------------------------
# Dynamic import of the training script to get model class
# -------------------------------------------------------

def load_model_class(train_script_path: str):
    spec = importlib.util.spec_from_file_location("train_gpt2_hetxl_module", train_script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GPT2HETXLCausalLM


# -------------------------------------------------------
# Simple text divergence metrics
# -------------------------------------------------------

def normalized_levenshtein(a: List[str], b: List[str]) -> float:
    """Token-level normalized edit distance in [0, 1]."""
    if len(a) == 0 and len(b) == 0:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost,  # substitute
            )
    dist = dp[len(a)][len(b)]
    return dist / max(len(a), len(b), 1)


def jaccard_distance(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    inter = sa & sb
    return 1.0 - (len(inter) / len(union))


def average_pairwise_metrics(texts: List[str]) -> Tuple[float, float]:
    tokenized = [t.strip().split() for t in texts]
    if len(tokenized) < 2:
        return 0.0, 0.0

    edit_vals = []
    jac_vals = []
    for i, j in combinations(range(len(tokenized)), 2):
        edit_vals.append(normalized_levenshtein(tokenized[i], tokenized[j]))
        jac_vals.append(jaccard_distance(tokenized[i], tokenized[j]))

    return sum(edit_vals) / len(edit_vals), sum(jac_vals) / len(jac_vals)


def distinct_fraction(texts: List[str]) -> float:
    uniq = len(set(t.strip() for t in texts))
    return uniq / max(len(texts), 1)


# -------------------------------------------------------
# Sampling helpers
# -------------------------------------------------------

def top_p_sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.95) -> torch.Tensor:
    """
    logits: [1, vocab]
    returns next token id tensor of shape [1, 1]
    """
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # keep smallest prefix with cumulative prob >= top_p
    mask = cumulative > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    next_sorted = torch.multinomial(sorted_probs, num_samples=1)
    next_token = torch.gather(sorted_idx, dim=-1, index=next_sorted)
    return next_token


# -------------------------------------------------------
# Span-level latent sampling
# -------------------------------------------------------

@torch.no_grad()
def sample_span_latent(model, hidden_last: torch.Tensor) -> torch.Tensor:
    """
    hidden_last: [1, D]
    returns latent epsilon: [1, D]

    Uses one heteroscedastic sample for the whole rollout span.
    """
    h = hidden_last.float()  # stability

    diag_logvar = model.diag_head(h)
    diag_logvar = torch.clamp(diag_logvar, min=-12.0, max=-4.0)

    gate = torch.sigmoid(model.noise_gate(h))
    gate = torch.clamp(gate, 0.0, 0.25)

    diag_std = torch.exp(0.5 * diag_logvar) * gate
    diag_std = torch.clamp(diag_std, max=0.1)
    diag_noise = diag_std * torch.randn_like(h)

    if getattr(model, "lowrank_head", None) is not None and model.lowrank_head is not None and model.hetxl_rank > 0:
        lowrank = model.lowrank_head(h).reshape(1, h.size(-1), model.hetxl_rank)
        lowrank = 0.05 * torch.tanh(lowrank)
        z = torch.randn(1, model.hetxl_rank, device=h.device, dtype=h.dtype)
        lowrank_noise = torch.einsum("bdr,br->bd", lowrank, z)
        lowrank_noise = gate * lowrank_noise
        lowrank_noise = torch.clamp(lowrank_noise, min=-0.1, max=0.1)
    else:
        lowrank_noise = torch.zeros_like(h)

    eps = diag_noise + lowrank_noise
    eps = torch.clamp(eps, min=-0.1, max=0.1)
    return eps


@torch.no_grad()
def generate_with_fixed_span_latent(
    model,
    tokenizer,
    prompt: str,
    span_latent: torch.Tensor,
    max_new_tokens: int = 24,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str = "cuda",
):
    """
    Generate by reusing the SAME latent perturbation for each generated step.
    This is a crude but useful approximation to "one interpretation per chain".
    """
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        outputs = model.transformer(input_ids=generated, return_dict=True)
        hidden = outputs.last_hidden_state[:, -1, :]  # [1, D]

        # Reuse fixed span latent for the whole chain
        noisy_hidden = hidden.float() + span_latent.float()

        temp = torch.exp(model.logit_temperature.float()).clamp(min=0.8, max=1.25)
        logits = F.linear(noisy_hidden, model.lm_head.weight.float()) / temp
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        next_token = top_p_sample(logits, temperature=temperature, top_p=top_p)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    completion = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text
    return completion


@torch.no_grad()
def get_prompt_latents(model, tokenizer, prompt: str, k: int, device: str):
    """
    Compute final prompt hidden state once, then sample k span latents from it.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.transformer(input_ids=inputs["input_ids"], return_dict=True)
    hidden_last = outputs.last_hidden_state[:, -1, :]  # [1, D]
    latents = [sample_span_latent(model, hidden_last) for _ in range(k)]
    return latents

@torch.no_grad()
def inspect_prompt_noise(model, tokenizer, prompt: str, device: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.transformer(input_ids=inputs["input_ids"], return_dict=True)
    h = outputs.last_hidden_state[:, -1, :].float()  # [1, D]

    diag_logvar = model.diag_head(h)
    diag_logvar = torch.clamp(diag_logvar, min=-12.0, max=-4.0)
    diag_std = torch.exp(0.5 * diag_logvar)

    gate = torch.sigmoid(model.noise_gate(h))

    stats = {
        "gate_mean": gate.mean().item(),
        "gate_max": gate.max().item(),
        "diag_std_mean": diag_std.mean().item(),
        "diag_std_max": diag_std.max().item(),
    }

    if getattr(model, "lowrank_head", None) is not None and model.lowrank_head is not None and model.hetxl_rank > 0:
        lowrank = model.lowrank_head(h).reshape(1, h.size(-1), model.hetxl_rank)
        lowrank = 0.05 * torch.tanh(lowrank)
        stats["lowrank_frob"] = lowrank.norm().item()
        stats["lowrank_abs_mean"] = lowrank.abs().mean().item()
    else:
        stats["lowrank_frob"] = 0.0
        stats["lowrank_abs_mean"] = 0.0

    # sample epsilon a few times
    eps_norms = []
    for _ in range(8):
        eps = sample_span_latent(model, h)
        eps_norms.append(eps.norm(dim=-1).mean().item())

    stats["eps_norm_mean"] = sum(eps_norms) / len(eps_norms)
    stats["eps_norm_max"] = max(eps_norms)

    return stats


# -------------------------------------------------------
# Prompt set
# -------------------------------------------------------

DEFAULT_PROMPTS = [
    ("clear", "The capital of France is"),
    ("ambiguous", "I went to the bank because I needed"),
    ("clear", "A triangle has three"),
    ("ambiguous", "The chicken is ready to eat"),
    ("clear", "The opposite of hot is"),
    ("ambiguous", "John told Bill that he was late because"),
    ("clear", "The sun rises in the"),
    ("ambiguous", "Put the trophy in the suitcase because it was too"),
]


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--train_script_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--t", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    GPT2HETXLCausalLM = load_model_class(args.train_script_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config.from_pretrained(args.model_dir)
    model = GPT2HETXLCausalLM.from_pretrained(args.model_dir, config=config)
    model.to(args.device)
    model.eval()

    print("=" * 100)
    print(f"Loaded model from: {args.model_dir}")
    print(f"Device: {args.device}")
    print(f"k={args.k}, t={args.t}, temperature={args.temperature}, top_p={args.top_p}")
    print("=" * 100)

    grouped = {"clear": [], "ambiguous": []}

    for label, prompt in DEFAULT_PROMPTS:
        latents = get_prompt_latents(model, tokenizer, prompt, args.k, args.device)

        completions = []
        for latent in latents:
            comp = generate_with_fixed_span_latent(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                span_latent=latent,
                max_new_tokens=args.t,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )
            completions.append(comp)

        edit_div, jac_div = average_pairwise_metrics(completions)
        distinct = distinct_fraction(completions)

        grouped[label].append((prompt, distinct, edit_div, jac_div, completions))

        print()
        print("-" * 100)
        print(f"[{label.upper()}] PROMPT: {prompt}")
        print(f"distinct_fraction={distinct:.3f}  avg_edit_div={edit_div:.3f}  avg_jaccard_div={jac_div:.3f}")
        print("-" * 100)
        for i, comp in enumerate(completions, 1):
            print(f"{i:02d}. {comp}")
        stats = inspect_prompt_noise(model, tokenizer, prompt, args.device)
        print(stats)
        print("-" * 100)

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for label in ["clear", "ambiguous"]:
        ds = [x[1] for x in grouped[label]]
        es = [x[2] for x in grouped[label]]
        js = [x[3] for x in grouped[label]]
        print(
            f"{label:>10} | distinct={sum(ds)/len(ds):.3f} | "
            f"edit_div={sum(es)/len(es):.3f} | jaccard_div={sum(js)/len(js):.3f}"
        )

    print()
    print("Heuristic read:")
    print("- Higher values on ambiguous prompts than clear prompts are a good first sign.")
    print("- If both clear and ambiguous are equally diverse, the latent may just be injecting generic noise.")
    print("- If both are nearly identical, the heteroscedastic branch may be too weak or ignored.")


if __name__ == "__main__":
    main()