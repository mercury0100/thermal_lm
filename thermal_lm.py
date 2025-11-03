# thermal_lm.py
"""
Thermal Language Models (Thermal LM)
------------------------------------

A drop-in Hugging Face / PyTorch wrapper that augments any decoder-only
Causal LM (e.g., GPT-2, OPT, etc.) with a *per-token learned temperature field*:

    τ_t = exp(logτ_t) = exp(fc_logtau(h_t))

The hidden temperature τ_t rescales the token-level hidden state before the LM head:

    logits_t = lm_head(h_t / τ_t)

Unlike post-hoc temperature scaling, τ_t is learned end-to-end as a contextual
control variable that modulates the energy–entropy balance of prediction.

Free-Energy Objective
---------------------
Training minimizes an approximate free energy per token:

    𝓛 = CE(logits, labels)
        − α · τ_t · H(p_θ(x_t | x_<t))
        + λ · ||logτ_t||²

where:
  • CE is cross-entropy (energy term),
  • H(p) is the predictive entropy (entropy term),
  • α controls the entropy-reward strength (≈ 1e-3 – 1e-2),
  • λ regularizes logτ to stay near 0 (τ ≈ 1).

This yields interpretable, context-adaptive temperature dynamics:
  - low τ (“cold”) in deterministic or syntactic regions,
  - high τ (“hot”) near semantic or structural uncertainty.

Key features
------------
• Works as a drop-in replacement for `AutoModelForCausalLM`
• Fully compatible with HF `Trainer` and `.generate()`
• Exposes per-token τ and logτ via `output.aux`
• Free-energy loss reflects thermodynamic structure of language
"""

from typing import Optional, Tuple, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    model_type = "thermal_lm"

    def __init__(
        self,
        base_model_name_or_path: str = "distilgpt2",
        tau_clamp_min: float = -3.0,
        tau_clamp_max: float = 2.0,
        lambda_reg: float = 1e-3,
        alpha_entropy: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.tau_clamp_min = float(tau_clamp_min)
        self.tau_clamp_max = float(tau_clamp_max)
        self.lambda_reg = float(lambda_reg)
        self.alpha_entropy = float(alpha_entropy)

        # ensure generation-friendly config flags
        self.is_encoder_decoder = False
        self.is_decoder = True


# =============================
# Model
# =============================

class ThermalLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ThermalLMConfig

    def __init__(self, config: ThermalLMConfig):
        super().__init__(config)

        # 1. Load base LM
        self.base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

        # 2. Extract transformer + head
        if hasattr(self.base, "transformer"):
            self.transformer = self.base.transformer
            self.lm_head = self.base.lm_head
            hidden = getattr(self.base.config, "n_embd", None)
        elif hasattr(self.base, "model"):
            self.transformer = self.base.model
            self.lm_head = self.base.get_output_embeddings()
            hidden = getattr(self.base.config, "hidden_size", None)
        else:
            self.transformer = None
            self.lm_head = self.base.get_output_embeddings()
            hidden = getattr(self.lm_head, "in_features", None)

        if hidden is None:
            raise RuntimeError("Could not infer hidden size for base LM.")

        # 3. Per-token log-temperature head
        self.fc_logtau = nn.Linear(hidden, 1)
        nn.init.normal_(self.fc_logtau.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc_logtau.bias, 0.0)

        # 4. Hyperparameters
        self.tau_clamp_min = config.tau_clamp_min
        self.tau_clamp_max = config.tau_clamp_max
        self.lambda_reg = config.lambda_reg
        self.alpha_entropy = config.alpha_entropy

    # Convenience constructor
    @classmethod
    def from_base_pretrained(
        cls,
        base_model_name_or_path: str = "distilgpt2",
        tau_clamp_min: float = -3.0,
        tau_clamp_max: float = 2.0,
        lambda_reg: float = 1e-3,
        alpha_entropy: float = 1e-3,
        **kwargs,
    ):
        cfg = ThermalLMConfig(
            base_model_name_or_path=base_model_name_or_path,
            tau_clamp_min=tau_clamp_min,
            tau_clamp_max=tau_clamp_max,
            lambda_reg=lambda_reg,
            alpha_entropy=alpha_entropy,
            **kwargs,
        )
        return cls(cfg)

    # Core forward
    def _forward_hidden_tau(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        if hasattr(self.base, "transformer"):
            base_out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            h = base_out.last_hidden_state
        elif hasattr(self.base, "model"):
            base_out = self.base.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            h = base_out.last_hidden_state
        else:
            raise RuntimeError("Unsupported base model structure.")

        # ensure device consistency
        if self.fc_logtau.weight.device != h.device:
            self.fc_logtau.to(device=h.device, dtype=h.dtype)

        logtau = self.fc_logtau(h).squeeze(-1).clamp(self.tau_clamp_min, self.tau_clamp_max)  # (B,T)
        tau = logtau.exp().unsqueeze(-1)  # (B,T,1)

        logits = self.lm_head(h / tau)  # scale hidden state by τ
        return logits, logtau, base_out

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:

        logits, logtau, base_out = self._forward_hidden_tau(input_ids, attention_mask, **kwargs)

        loss = None
        if labels is not None:
            B, T, V = logits.size()
            # Cross-entropy per token
            ce = F.cross_entropy(
                logits.reshape(-1, V),
                labels.reshape(-1),
                reduction="none"
            ).reshape(B, T)

            # Predictive entropy per token (normalized)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = entropy / math.log(V)

            tau = logtau.exp()
            # Free-energy loss
            loss = (ce - self.alpha_entropy * tau * entropy).mean()
            loss = loss + self.lambda_reg * (logtau ** 2).mean()

        out = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=getattr(base_out, "past_key_values", None),
            hidden_states=getattr(base_out, "hidden_states", None),
            attentions=getattr(base_out, "attentions", None),
            cross_attentions=getattr(base_out, "cross_attentions", None),
        )
        setattr(out, "aux", {"logtau": logtau, "tau": logtau.exp()})
        return out