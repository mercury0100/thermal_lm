#!/usr/bin/env python3
"""
train_gpt2_hetxl.py

GPT-2 + HET-XL-style heteroscedastic heads for causal LM.

Features:
- Optional stochastic hidden-state perturbations before the LM head
- Low-rank + diagonal noise parameterization
- Monte Carlo marginal training objective
- KL-style regularizer toward a small isotropic prior
- Full-parameter training
- Safer fp32 stochastic path to avoid NaNs

Example baseline:
    python train_gpt2.py \
        --model_name gpt2 \
        --dataset_name wikitext \
        --output_dir ./gpt2-baseline-wikitext

Example noisy run:
    python train_gpt2.py \
        --model_name gpt2 \
        --dataset_name wikitext \
        --output_dir ./gpt2-hetxl-kl \
        --use_noise \
        --rank 2 \
        --mc_train_samples 1 \
        --mc_eval_samples 1 \
        --prior_std 0.05 \
        --beta_diag_kl 1e-3 \
        --beta_lowrank_kl 1e-3

Tiny C4 run:
    python train_gpt2.py \
      --model_name gpt2 \
      --dataset_name c4 \
      --c4_train_examples 200000 \
      --c4_val_examples 10000 \
      --output_dir ./gpt2-hetxl-c4 \
      --use_noise \
      --rank 4 \
      --mc_train_samples 1 \
      --mc_eval_samples 1 \
      --prior_std 0.05 \
      --beta_diag_kl 1e-3 \
      --beta_lowrank_kl 1e-3
"""

import argparse
import math
import os
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2PreTrainedModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class GPT2HETXLCausalLM(GPT2PreTrainedModel):
    """
    GPT-2 + heteroscedastic hidden-state noise heads.

    Predicts:
      diag_logvar: [B, T, D]
      lowrank:     [B, T, D, R]  (if rank > 0)

    Noise sample:
      eps = diag_std * z1 + einsum(lowrank, z2)
      z1 ~ N(0, I_D)
      z2 ~ N(0, I_R)

    We inject eps before the tied LM head.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.transformer = GPT2LMHeadModel(config).transformer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.hetxl_rank = int(getattr(config, "hetxl_rank", 8))
        self.init_logvar = float(getattr(config, "hetxl_init_logvar", -6.0))
        self.prior_std = float(getattr(config, "hetxl_prior_std", 0.05))
        self.beta_diag_kl = float(getattr(config, "hetxl_beta_diag_kl", 1e-4))
        self.beta_lowrank_kl = float(getattr(config, "hetxl_beta_lowrank_kl", 1e-4))

        hidden = config.n_embd

        self.diag_head = nn.Linear(hidden, hidden)

        if self.hetxl_rank > 0:
            self.lowrank_head = nn.Linear(hidden, hidden * self.hetxl_rank)
        else:
            self.lowrank_head = None

        self.noise_gate = nn.Linear(hidden, 1)
        self.logit_temperature = nn.Parameter(torch.tensor(0.0))

        self.post_init()

        nn.init.zeros_(self.diag_head.weight)
        nn.init.constant_(self.diag_head.bias, self.init_logvar)

        if self.lowrank_head is not None:
            nn.init.zeros_(self.lowrank_head.weight)
            nn.init.zeros_(self.lowrank_head.bias)

        nn.init.zeros_(self.noise_gate.weight)
        nn.init.constant_(self.noise_gate.bias, -6.0)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _sample_hidden_noise(
        self,
        hidden_states: torch.Tensor,
        diag_logvar: torch.Tensor,
        lowrank: Optional[torch.Tensor],
        gate: torch.Tensor,
    ) -> torch.Tensor:
        diag_std = torch.exp(0.5 * diag_logvar) * gate
        diag_std = torch.clamp(diag_std, min=0.0, max=0.03)

        z1 = torch.randn_like(hidden_states)
        diag_noise = diag_std * z1

        if lowrank is not None:
            z2 = torch.randn(
                hidden_states.size(0),
                hidden_states.size(1),
                self.hetxl_rank,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            lowrank_noise = torch.einsum("btdr,btr->btd", lowrank, z2)
            lowrank_noise = gate * lowrank_noise
            lowrank_noise = torch.clamp(lowrank_noise, min=-0.06, max=0.06)
        else:
            lowrank_noise = torch.zeros_like(hidden_states)

        eps = diag_noise + lowrank_noise
        return eps
        
    def _compute_kl_regularizer(
        self,
        diag_logvar: torch.Tensor,
        lowrank: Optional[torch.Tensor],
        gate: torch.Tensor,
    ) -> torch.Tensor:
        prior_var = self.prior_std ** 2

        diag_var = torch.exp(diag_logvar)
        eff_diag_var = (gate ** 2) * diag_var
        eff_diag_var = torch.clamp(eff_diag_var, min=1e-8)

        # 1. Diagonal KL (Trace + Entropy)
        ratio = eff_diag_var / prior_var
        kl_diag = 0.5 * (ratio - 1.0 - torch.log(ratio))
        kl_diag = kl_diag.mean()

        if lowrank is not None:
            R = self.hetxl_rank
            D_dim = diag_logvar.size(-1)
            gated_lowrank = gate.unsqueeze(-1) * lowrank  # [B, T, D, R]
            
            # 2. Low-Rank Trace Penalty
            # tr(U U^T) / prior_var. Scaled by 1/D to match kl_diag.mean()
            trace_term = (gated_lowrank.pow(2).sum(dim=-1) / prior_var).mean()
            
            # 3. Low-Rank Entropy Reward (Matrix Determinant Lemma)
            # ln |I + U^T D^{-1} U|
            D_inv = 1.0 / eff_diag_var # [B, T, D]
            
            # inner_mat shape: [B, T, R, R]
            inner_mat = torch.einsum("btdr,btd,btdk->btrk", gated_lowrank, D_inv, gated_lowrank)
            I_R = torch.eye(R, device=inner_mat.device, dtype=inner_mat.dtype)
            inner_mat = inner_mat + I_R
            
            # Compute log determinant (slogdet is safer for gradients)
            sign, logdet = torch.linalg.slogdet(inner_mat)
            logdet = torch.where(sign > 0, logdet, torch.zeros_like(logdet))
            
            # Scale down by D_dim to match the per-dimension scale of kl_diag
            logdet_reward = logdet.mean() / D_dim
            
            # Full low-rank KL contribution
            kl_lowrank = 0.5 * (trace_term - logdet_reward)
        else:
            kl_lowrank = diag_logvar.new_zeros(())

        kl = self.beta_diag_kl * kl_diag + self.beta_lowrank_kl * kl_lowrank
        return kl

    def _compute_mc_marginal_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        mc_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          loss, mean_logits

        MC marginalization:
          log p(y|x) ≈ logmeanexp_s log p(y|x, eps_s)

        Loss is normalized to per-token scale.
        """
        B, T, D = hidden_states.shape

        autocast_ctx = (
            torch.autocast(device_type=hidden_states.device.type, enabled=False)
            if hidden_states.device.type in {"cuda", "cpu"}
            else nullcontext()
        )

        with autocast_ctx:
            hidden_states_fp32 = hidden_states.float()
            labels = labels.long()

            diag_logvar = self.diag_head(hidden_states_fp32)
            diag_logvar = torch.clamp(diag_logvar, min=-12.0, max=-2.5)

            if self.lowrank_head is not None:
                lowrank = self.lowrank_head(hidden_states_fp32).reshape(B, T, D, self.hetxl_rank)
            else:
                lowrank = None

            gate = torch.sigmoid(self.noise_gate(hidden_states_fp32))
            gate = torch.clamp(gate, min=0.0, max=0.20)

            kl_reg = self._compute_kl_regularizer(
                diag_logvar=diag_logvar,
                lowrank=lowrank,
                gate=gate,
            )

            temp = torch.exp(self.logit_temperature.float()).clamp(min=0.9, max=1.1)

            logits_samples = []
            seq_logprobs = []

            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels.ne(-100)

            lm_weight = self.lm_head.weight.float()

            for _ in range(mc_samples):
                eps = self._sample_hidden_noise(hidden_states_fp32, diag_logvar, lowrank, gate)
                noisy_hidden = hidden_states_fp32 + eps

                logits = F.linear(noisy_hidden, lm_weight) / temp
                logits = torch.clamp(logits, min=-30.0, max=30.0)
                logits_samples.append(logits)

                shift_logits = logits[..., :-1, :].contiguous()
                log_probs = F.log_softmax(shift_logits, dim=-1)

                gather_labels = shift_labels.masked_fill(~valid_mask, 0).unsqueeze(-1)
                selected = torch.gather(log_probs, dim=-1, index=gather_labels).squeeze(-1)
                selected = selected.masked_fill(~valid_mask, 0.0)

                token_count = valid_mask.sum(dim=-1).clamp(min=1)
                seq_logprob = selected.sum(dim=-1) / token_count
                seq_logprobs.append(seq_logprob)

            seq_logprobs = torch.stack(seq_logprobs, dim=0)
            seq_log_marginal = torch.logsumexp(seq_logprobs, dim=0) - math.log(mc_samples)

            lm_loss = -seq_log_marginal.mean()
            loss = lm_loss + kl_reg

            if not torch.isfinite(diag_logvar).all():
                raise RuntimeError("Non-finite diag_logvar")
            if lowrank is not None and not torch.isfinite(lowrank).all():
                raise RuntimeError("Non-finite lowrank")
            if not torch.isfinite(gate).all():
                raise RuntimeError("Non-finite gate")
            if not torch.isfinite(lm_loss):
                raise RuntimeError(f"Non-finite lm_loss: {lm_loss.item()}")
            if not torch.isfinite(kl_reg):
                raise RuntimeError(f"Non-finite kl_reg: {kl_reg.item()}")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss in _compute_mc_marginal_loss")

            mean_logits = torch.stack(logits_samples, dim=0).mean(dim=0).to(hidden_states.dtype)
            return loss, mean_logits

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        mc_samples: Optional[int] = None,
        use_noise: bool = False,
    ):
        mc_samples = mc_samples or int(getattr(self.config, "hetxl_mc_train_samples", 2))

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = transformer_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        loss = None

        if use_noise and labels is not None:
            loss, logits = self._compute_mc_marginal_loss(
                hidden_states=hidden_states,
                labels=labels,
                mc_samples=mc_samples,
            )
        elif labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class HETXLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            mc_samples=self.args.mc_train_samples,
            use_noise=self.args.use_noise,
            return_dict=True,
        )
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected in training: {loss}")
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = inputs.get("labels") is not None

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
                mc_samples=self.args.mc_eval_samples,
                use_noise=self.args.use_noise,
                return_dict=True,
            )

        loss = outputs.loss.detach() if has_labels else None

        if prediction_loss_only:
            return (loss, None, None)

        logits = outputs.logits.detach()
        labels = inputs.get("labels")
        return (loss, logits, labels)


def get_text_column(dataset):
    for c in ["text", "content"]:
        if c in dataset.column_names:
            return c
    raise ValueError(f"Could not find a text column in {dataset.column_names}")


def tokenize_and_group(dataset_dict, tokenizer, block_size):
    text_column = get_text_column(dataset_dict["train"])

    def tokenize_fn(examples):
        return tokenizer(examples[text_column])

    tokenized = dataset_dict.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing",
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = [x.copy() for x in result["input_ids"]]
        return result

    lm_datasets = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Grouping into blocks of {block_size}",
    )
    return lm_datasets


def load_lm_dataset(args):
    if args.dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        if isinstance(ds, DatasetDict):
            return ds
        raise ValueError("Expected DatasetDict for wikitext")
    elif args.dataset_name == "c4":
        train = load_dataset("c4", "en", split=f"train[:{args.c4_train_examples}]")
        val = load_dataset("c4", "en", split=f"validation[:{args.c4_val_examples}]")
        return DatasetDict({"train": train, "validation": val})
    else:
        raise ValueError("dataset_name must be one of: wikitext, c4")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="wikitext", choices=["wikitext", "c4"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--c4_train_examples", type=int, default=20000)
    parser.add_argument("--c4_val_examples", type=int, default=2000)

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--init_logvar", type=float, default=-6.0)
    parser.add_argument("--mc_train_samples", type=int, default=2)
    parser.add_argument("--mc_eval_samples", type=int, default=4)
    parser.add_argument("--use_noise", action="store_true")
    parser.add_argument("--prior_std", type=float, default=0.05)
    parser.add_argument("--beta_diag_kl", type=float, default=1e-4)
    parser.add_argument("--beta_lowrank_kl", type=float, default=1e-4)

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = load_lm_dataset(args)
    lm_datasets = tokenize_and_group(raw_datasets, tokenizer, args.block_size)

    config = GPT2Config.from_pretrained(args.model_name)
    config.pad_token_id = tokenizer.pad_token_id
    config.hetxl_rank = args.rank
    config.hetxl_init_logvar = args.init_logvar
    config.hetxl_mc_train_samples = args.mc_train_samples
    config.hetxl_mc_eval_samples = args.mc_eval_samples
    config.hetxl_prior_std = args.prior_std
    config.hetxl_beta_diag_kl = args.beta_diag_kl
    config.hetxl_beta_lowrank_kl = args.beta_lowrank_kl

    model = GPT2HETXLCausalLM.from_pretrained(args.model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    training_args.mc_train_samples = args.mc_train_samples
    training_args.mc_eval_samples = args.mc_eval_samples
    training_args.use_noise = args.use_noise

    print(f"Noise enabled: {args.use_noise}")
    print(
        f"prior_std={args.prior_std} "
        f"beta_diag_kl={args.beta_diag_kl} "
        f"beta_lowrank_kl={args.beta_lowrank_kl}"
    )

    trainer = HETXLTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    try:
        eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        eval_metrics["perplexity"] = float("inf")

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("\nDone.")
    print(f"Noise enabled: {args.use_noise}")
    print(f"Model saved to: {args.output_dir}")
    print(f"Eval loss: {eval_metrics['eval_loss']:.4f}")
    print(f"Perplexity: {eval_metrics['perplexity']:.4f}")


if __name__ == "__main__":
    main()