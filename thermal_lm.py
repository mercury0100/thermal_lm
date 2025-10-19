# thermal_lm.py
"""
Thermal Language Models (Thermal LM)
------------------------------------

A drop-in Hugging Face / PyTorch wrapper that adds a **per-token hidden temperature**
τ(x) = exp(logτ) to any decoder-only Causal LM (e.g., GPT-2 family). We scale the
final hidden states before the LM head:

    logits_t = lm_head( h_t / τ_t ),  where logτ_t = fc_logtau(h_t)

This yields a deterministic, interpretable proxy for **aleatoric uncertainty** while
remaining fully compatible with HF `Trainer` and `.generate()`.

Key features
- `save_pretrained` / `from_pretrained` work as usual
- Returns standard `CausalLMOutputWithCrossAttentions`
- Exposes per-token `logtau` in `output.aux["logtau"]`
- Inherits `GenerationMixin` so `.generate()` keeps working on >= v4.50

Typical use
-----------
from thermal_lm import ThermalLMForCausalLM

model = ThermalLMForCausalLM.from_base_pretrained(
    "distilgpt2", tau_clamp_min=-3.0, tau_clamp_max=2.0, lambda_reg=1e-3
)
out = model(input_ids=..., attention_mask=..., labels=...)
loss = out.loss
logtau = out.aux["logtau"]
"""

from typing import Optional, Tuple, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


# =============================
# Config
# =============================

class ThermalLMConfig(PretrainedConfig):
    """
    Minimal wrapper config for Thermal LM.
    We do NOT merge the base LM's config here. We just record:
      - which base model to wrap
      - the thermal head hyperparameters
    """
    model_type = "thermal_lm"

    def __init__(
        self,
        base_model_name_or_path: str = "distilgpt2",
        tau_clamp_min: float = -3.0,
        tau_clamp_max: float = 2.0,
        lambda_reg: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.tau_clamp_min = float(tau_clamp_min)
        self.tau_clamp_max = float(tau_clamp_max)
        self.lambda_reg = float(lambda_reg)

        # generation-friendly flags (decoder-only)
        self.is_encoder_decoder = False
        self.is_decoder = True


# =============================
# Model
# =============================

class ThermalLMForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Thermal LM: per-token hidden temperature modulation for causal LMs.

    Forward returns a *standard* CausalLMOutputWithCrossAttentions:
      - .loss  : CE(logits, labels) + λ * mean(logτ)  (if labels provided)
      - .logits
      - .past_key_values / .hidden_states / .attentions / .cross_attentions (if provided by base)
      - .aux   : dict with {"logtau": (B,T)} for downstream inspection/visualization

    `.generate()` works because we inherit GenerationMixin and return standard outputs.
    """
    config_class = ThermalLMConfig

    def __init__(self, config: ThermalLMConfig):
        super().__init__(config)

        # 1) Load base decoder-only LM (e.g., GPT-2/OPT/etc.)
        self.base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

        # 2) Get references to transformer and lm_head (GPT-2 style if available)
        #    We try common patterns; this keeps things simple without hard dependencies.
        if hasattr(self.base, "transformer"):                 # GPT-2 like
            self.transformer = self.base.transformer
            self.lm_head = self.base.lm_head
            hidden = getattr(self.base.config, "n_embd", None)
        elif hasattr(self.base, "model"):                      # Some models expose .model
            self.transformer = self.base.model
            self.lm_head = self.base.get_output_embeddings()
            hidden = getattr(self.base.config, "hidden_size", None)
        else:
            # Fall back to output embedding inference if available
            self.transformer = None  # we will attempt to call self.base.model in _forward_hidden_tau if present
            self.lm_head = self.base.get_output_embeddings()
            hidden = getattr(self.lm_head, "in_features", None)

        if hidden is None:
            raise RuntimeError(
                "Could not infer hidden size for the thermal head. "
                "This wrapper currently supports GPT-2-like causal LMs (with .transformer/.lm_head) "
                "or models exposing .model and output embeddings."
            )

        # 3) Per-token log-temperature head
        self.fc_logtau = nn.Linear(hidden, 1)
        nn.init.normal_(self.fc_logtau.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc_logtau.bias, 0.0)  # τ≈1 initially

        # 4) Hyperparameters
        self.tau_clamp_min = config.tau_clamp_min
        self.tau_clamp_max = config.tau_clamp_max
        self.lambda_reg = config.lambda_reg

    # -------- Convenience constructor --------
    @classmethod
    def from_base_pretrained(
        cls,
        base_model_name_or_path: str = "distilgpt2",
        tau_clamp_min: float = -3.0,
        tau_clamp_max: float = 2.0,
        lambda_reg: float = 1e-3,
        **kwargs,
    ):
        cfg = ThermalLMConfig(
            base_model_name_or_path=base_model_name_or_path,
            tau_clamp_min=tau_clamp_min,
            tau_clamp_max=tau_clamp_max,
            lambda_reg=lambda_reg,
            **kwargs,
        )
        return cls(cfg)

    # -------- (Optional) Embedding helpers to keep weight-tying clean --------
    def get_input_embeddings(self):
        if hasattr(self.base, "get_input_embeddings"):
            return self.base.get_input_embeddings()
        # GPT-2 style
        if hasattr(self.base, "transformer"):
            return self.base.transformer.wte
        raise AttributeError("Base model does not expose input embeddings")

    def set_input_embeddings(self, value):
        if hasattr(self.base, "set_input_embeddings"):
            self.base.set_input_embeddings(value)
            return
        if hasattr(self.base, "transformer"):
            self.base.transformer.wte = value
            return
        raise AttributeError("Base model does not allow setting input embeddings")

    def get_output_embeddings(self):
        if hasattr(self.base, "get_output_embeddings"):
            return self.base.get_output_embeddings()
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if hasattr(self.base, "set_output_embeddings"):
            self.base.set_output_embeddings(new_embeddings)
        else:
            self.lm_head = new_embeddings

    # -------- Generation plumbing --------
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Prefer delegating to the base for maximum compatibility (KV cache, etc.)
        if hasattr(self.base, "prepare_inputs_for_generation"):
            return self.base.prepare_inputs_for_generation(input_ids, **kwargs)
        # Minimal fallback
        return {"input_ids": input_ids, **kwargs}

    def _reorder_cache(self, past_key_values, beam_idx):
        if hasattr(self.base, "_reorder_cache"):
            return self.base._reorder_cache(past_key_values, beam_idx)
        # Default behavior
        return super()._reorder_cache(past_key_values, beam_idx)

    # -------- Core forward --------
    def _forward_hidden_tau(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Returns:
          logits:  (B,T,V) tempered logits
          logtau:  (B,T)   per-token log temperature (clamped)
          base_out: the base transformer outputs (to pass through caches/attn/etc.)
        """
        # Prefer GPT-2-style transformer if present:
        if hasattr(self.base, "transformer"):
            base_out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            h = base_out.last_hidden_state  # (B,T,H)
        elif hasattr(self.base, "model"):
            # Some models expose .model(...)->BaseModelOutputWithPast containing last_hidden_state
            base_out = self.base.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            h = base_out.last_hidden_state
        else:
            raise RuntimeError(
                "Unsupported base model structure for Thermal LM wrapper. "
                "Expected `.transformer` or `.model` with `last_hidden_state`."
            )

        logtau = self.fc_logtau(h).squeeze(-1).clamp(self.tau_clamp_min, self.tau_clamp_max)  # (B,T)
        tau = logtau.exp().unsqueeze(-1)                                                      # (B,T,1)
        logits = self.lm_head(h / tau)                                                        # (B,T,V)
        return logits, logtau, base_out

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        """
        Standard causal LM forward.
        If `labels` are provided:
           loss = CE(logits, labels) + lambda_reg * mean(logtau)

        Returns CausalLMOutputWithCrossAttentions, with an extra `aux` field:
           output.aux = {"logtau": (B,T)}
        """
        logits, logtau, base_out = self._forward_hidden_tau(input_ids, attention_mask, **kwargs)

        loss = None
        if labels is not None:
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), reduction="mean")
            loss = loss + (self.lambda_reg * (logtau ** 2).mean())

        out = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=getattr(base_out, "past_key_values", None),
            hidden_states=getattr(base_out, "hidden_states", None),
            attentions=getattr(base_out, "attentions", None),
            cross_attentions=getattr(base_out, "cross_attentions", None),
        )
        # Attach auxiliary τ (not used by `generate`, available to user code)
        setattr(out, "aux", {"logtau": logtau})
        return out


# =============================
# Optional quick smoke test
# =============================
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok_name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(tok_name)
    model = ThermalLMForCausalLM.from_base_pretrained(tok_name, lambda_reg=1e-3).eval()

    inputs = tok("The thermodynamics of language suggest that", return_tensors="pt")
    out = model(**inputs)
    print("logits:", tuple(out.logits.shape))
    print("logtau:", tuple(out.aux["logtau"].shape))

    gen = model.generate(**inputs, max_new_tokens=24, do_sample=True, top_p=0.95)
    print(tok.decode(gen[0], skip_special_tokens=True))