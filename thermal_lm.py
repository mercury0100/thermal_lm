"""
Thermal Language Models (Thermal LM)
------------------------------------

A drop-in Hugging Face / PyTorch wrapper that augments any decoder-only
Causal LM (e.g., GPT-2, OPT, etc.) with a *per-token learned temperature field*:

    τ_t = exp(logτ_t) = exp(fc_logtau(h_t))

The hidden temperature τ_t rescales a *normalised* token-level hidden state
before the LM head:

    ĥ_t   = h_t / ||h_t||_2
    logits_t = lm_head(ĥ_t / τ_t)

Unlike post-hoc temperature scaling, τ_t is learned end-to-end as a contextual
control variable that modulates the energy–entropy balance of prediction.

Training Objective (no thermodynamic penalty)
---------------------------------------------
We train with standard token-level cross-entropy on τ-scaled logits:

    CE_t(τ_t) = -log Softmax(logits_t)[y_t],
    logits_t = lm_head(ĥ_t / τ_t),

plus an optional quadratic regulariser on logτ_t:

    𝓛_t = CE_t(τ_t) + λ · ||logτ_t||²

where:
  • τ_t is predicted from the hidden state h_t,
  • ĥ_t is a per-token L2-normalised hidden state,
  • λ regularizes logτ to stay near 0 (τ ≈ 1 on average).

Generation uses the same τ-scaled logits: lm_head(ĥ_t / τ_t).
"""

from __future__ import annotations

from typing import Optional, Any

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
    model_type = "thermal_lm"

    def __init__(
        self,
        base_model_name_or_path: str = "distilgpt2",
        tau_clamp_min: float = -3.0,
        tau_clamp_max: float = 2.0,
        lambda_reg: float = 0.0,
        tau_smooth: bool = False,
        **kwargs,
    ):
        """
        Args:
            base_model_name_or_path: HF identifier for the base causal LM.
            tau_clamp_min: lower bound on log τ (hard clamp or tanh range).
            tau_clamp_max: upper bound on log τ.
            lambda_reg: coefficient on (logτ)^2 regularisation term.
            tau_smooth: if True, use tanh squashing into [min, max] instead
                        of hard clamp.
        """
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.tau_clamp_min = float(tau_clamp_min)
        self.tau_clamp_max = float(tau_clamp_max)
        self.lambda_reg = float(lambda_reg)
        self.tau_smooth = bool(tau_smooth)

        # ensure generation-friendly config flags
        self.is_encoder_decoder = False
        self.is_decoder = True


# =============================
# Model
# =============================

class ThermalLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ThermalLMConfig

    # Convenience constructor
    @classmethod
    def from_base_pretrained(
        cls,
        base_model_name_or_path: str = "distilgpt2",
        tau_clamp_min: float = -3.0,
        tau_clamp_max: float = 2.0,
        lambda_reg: float = 0.0,
        tau_smooth: bool = False,
        **kwargs,
    ):
        cfg = ThermalLMConfig(
            base_model_name_or_path=base_model_name_or_path,
            tau_clamp_min=tau_clamp_min,
            tau_clamp_max=tau_clamp_max,
            lambda_reg=lambda_reg,
            tau_smooth=tau_smooth,
            **kwargs,
        )
        return cls(cfg)

    def __init__(self, config: ThermalLMConfig):
        super().__init__(config)

        # 1) Load base LM
        self.base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path
        )

        # 2) Extract transformer + head + hidden size
        hidden = None
        if hasattr(self.base, "transformer"):
            self.transformer = self.base.transformer
            # many GPT-like models expose lm_head
            self.lm_head = getattr(self.base, "lm_head", self.base.get_output_embeddings())
            hidden = getattr(self.base.config, "n_embd", None)
        elif hasattr(self.base, "model"):
            self.transformer = self.base.model
            self.lm_head = self.base.get_output_embeddings()
            hidden = getattr(self.base.config, "hidden_size", None)
        else:
            # very generic fallback
            self.transformer = None
            self.lm_head = self.base.get_output_embeddings()
            hidden = getattr(self.lm_head, "in_features", None)

        if hidden is None:
            raise RuntimeError("Could not infer hidden size for base LM.")

        # 3) Per-token log-temperature head (scalar)
        self.fc_logtau = nn.Linear(hidden, 1)
        nn.init.normal_(self.fc_logtau.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc_logtau.bias, 0.0)

        # 4) Hyperparameters
        self.tau_clamp_min = config.tau_clamp_min
        self.tau_clamp_max = config.tau_clamp_max
        self.tau_smooth = config.tau_smooth
        self.lambda_reg = config.lambda_reg

        # training helpers
        self.ignore_index = -100  # HF default

    # -----------------------------
    # τ parameterization utilities
    # -----------------------------
    def _tau_from_logtau(self, logtau_raw: torch.Tensor):
        """
        Map raw logτ to (logτ_clamped_or_squashed, τ) with either:
          - hard clamp to [tau_clamp_min, tau_clamp_max], or
          - smooth tanh into that range if tau_smooth=True.
        """
        if self.tau_smooth:
            lo, hi = self.tau_clamp_min, self.tau_clamp_max
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            logtau = mid + half * torch.tanh(logtau_raw)
        else:
            logtau = logtau_raw.clamp(self.tau_clamp_min, self.tau_clamp_max)

        tau = torch.exp(logtau)
        return logtau, tau

    # -----------------------------
    # Backbone forward
    # -----------------------------
    def _forward_hidden(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Any,
    ):
        """
        Run the base transformer to get hidden states (h) and the original model output.
        """
        if hasattr(self.base, "transformer"):
            out = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            h = out.last_hidden_state
        else:
            out = self.base.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            h = out.last_hidden_state
        return h, out

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithCrossAttentions:

        # Base hidden states
        h, base_out = self._forward_hidden(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        # temperatures
        logtau_raw = self.fc_logtau(h).squeeze(-1)           # (B, T)
        logtau_full, tau_full = self._tau_from_logtau(logtau_raw)  # both (B, T)

        # -------------------------------------------
        # 1) Per-token L2 normalisation of hidden state
        # -------------------------------------------
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)   # (B, T, D)

        # -------------------------------------------
        # 2) Scale by τ: h_scaled = h_norm / τ
        # -------------------------------------------
        h_scaled = h_norm / tau_full.unsqueeze(-1)           # (B, T, D)

        # -------------------------------------------
        # 3) Project to logits
        # -------------------------------------------
        logits_tau = self.lm_head(h_scaled)                  # (B, T, V)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # shift for causal LM
            logits_data = logits_tau[:, :-1, :]              # (B, T-1, V)
            targets = labels[:, 1:]                          # (B, T-1)

            if attention_mask is not None:
                attn = attention_mask[:, 1:]
            else:
                attn = torch.ones_like(targets, dtype=torch.long)

            valid = (targets != self.ignore_index) & (attn > 0)

            # STANDARD CE ON τ-SCALED LOGITS
            logp = F.log_softmax(logits_data, dim=-1)        # (B, T-1, V)
            gathered = logp.gather(
                -1, targets.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)                                    # (B, T-1)
            ce_loss = -gathered[valid]                       # (N_valid,)

            loss = ce_loss.mean()

            # regularise logτ toward 0 (τ ≈ 1)
            logtau_valid = logtau_full[:, :-1][valid]
            if self.lambda_reg > 0.0 and logtau_valid.numel() > 0:
                loss = loss + self.lambda_reg * (logtau_valid ** 2).mean()

        out = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits_tau,  # generation uses τ-scaled logits
            past_key_values=getattr(base_out, "past_key_values", None),
            hidden_states=getattr(base_out, "hidden_states", None),
            attentions=getattr(base_out, "attentions", None),
            cross_attentions=getattr(base_out, "cross_attentions", None),
        )

        # Attach aux fields for analysis/visualization (unshifted length)
        out.aux = {"logtau": logtau_full, "tau": tau_full}
        return out

    # -----------------------------
    # Convenience utils
    # -----------------------------
    @torch.no_grad()
    def get_tau_map(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Return per-token τ for a batch without computing loss.
        Shape: (B, T).
        """
        h, _ = self._forward_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        logtau_raw = self.fc_logtau(h).squeeze(-1)
        _, tau = self._tau_from_logtau(logtau_raw)
        return tau
