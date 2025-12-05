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
# Config (unchanged)
# =============================

class ThermalLMConfig(PretrainedConfig):
    model_type = "thermal_lm"

    def __init__(
        self,
        base_model_name_or_path: str = "distilgpt2",
        logsigma_clamp_min: float = -6.0,
        logsigma_clamp_max: float = 2.0,
        lambda_reg: float = 1e-5,          # kept for backwards compat; not used
        sigma_smooth: bool = False,
        noise_at_eval: bool = False,
        num_mc_samples_train: int = 5,     # kept for backwards compat; not used
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.logsigma_clamp_min = float(logsigma_clamp_min)
        self.logsigma_clamp_max = float(logsigma_clamp_max)
        self.lambda_reg = float(lambda_reg)
        self.sigma_smooth = bool(sigma_smooth)
        self.noise_at_eval = bool(noise_at_eval)
        self.num_mc_samples_train = int(num_mc_samples_train)

        self.is_encoder_decoder = False
        self.is_decoder = True


# =============================
# Calibrated heteroscedastic LM
# =============================

class ThermalLMForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Variant of ThermalLM where σ behaves like calibrated aleatoric uncertainty.

    Key change vs the original:
      - Training loss (for valid next-token positions) is

            s_t      = 2 * logσ_t          # = log σ_t^2
            CE_t     = cross_entropy(z_t, y_t)
            L_t      = exp(-s_t) * CE_t + s_t

        So at optimum, exp(-s_t) * CE_t ≈ 1 => s_t ≈ log CE_t and
        σ_t^2 = exp(s_t) ≈ CE_t. Harder tokens → larger σ_t.

      - No MC over logit noise during training; noise is only used
        at inference if `stochastic=True` or `noise_at_eval=True`.
    """

    config_class = ThermalLMConfig

    # Convenience constructor
    @classmethod
    def from_base_pretrained(
        cls,
        base_model_name_or_path: str = "distilgpt2",
        logsigma_clamp_min: float = -6.0,
        logsigma_clamp_max: float = 2.0,
        lambda_reg: float = 1e-5,
        sigma_smooth: bool = False,
        noise_at_eval: bool = False,
        num_mc_samples_train: int = 5,
        **kwargs,
    ):
        cfg = ThermalLMConfig(
            base_model_name_or_path=base_model_name_or_path,
            logsigma_clamp_min=logsigma_clamp_min,
            logsigma_clamp_max=logsigma_clamp_max,
            lambda_reg=lambda_reg,
            sigma_smooth=sigma_smooth,
            noise_at_eval=noise_at_eval,
            num_mc_samples_train=num_mc_samples_train,
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
            self.lm_head = getattr(self.base, "lm_head", self.base.get_output_embeddings())
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

        # 3) Per-token log σ_t head (scalar diffusion / temperature scale)
        self.fc_logsigma = nn.Linear(hidden, 1)
        nn.init.normal_(self.fc_logsigma.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc_logsigma.bias, 0.0)

        # 4) Hyperparameters
        self.logsigma_clamp_min = config.logsigma_clamp_min
        self.logsigma_clamp_max = config.logsigma_clamp_max
        self.sigma_smooth = config.sigma_smooth
        self.noise_at_eval = config.noise_at_eval

        self.ignore_index = -100  # HF default

    # -----------------------------
    # σ parameterization utilities
    # -----------------------------
    def _sigma_from_logsigma(self, logsigma_raw: torch.Tensor):
        """
        Map raw log σ to (logsigma_clamped_or_squashed, σ).

        logsigma is log σ (not log σ^2). For the heteroscedastic loss we use
        s = 2 * logsigma = log σ^2.
        """
        if self.sigma_smooth:
            lo, hi = self.logsigma_clamp_min, self.logsigma_clamp_max
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            logsigma = mid + half * torch.tanh(logsigma_raw)
        else:
            logsigma = logsigma_raw.clamp(self.logsigma_clamp_min, self.logsigma_clamp_max)

        sigma = torch.exp(logsigma)
        return logsigma, sigma

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
        stochastic: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithCrossAttentions:

        # Decide whether to inject noise into logits for this call
        if stochastic is None:
            inject_noise_for_logits = self.training or self.noise_at_eval
        else:
            inject_noise_for_logits = bool(stochastic)

        # Base hidden states
        h, base_out = self._forward_hidden(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        # Per-token σ_t
        logsigma_raw = self.fc_logsigma(h).squeeze(-1)                        # (B, T)
        logsigma_full, sigma_full = self._sigma_from_logsigma(logsigma_raw)   # both (B, T)

        # Base logits z_t
        logits_base = self.lm_head(h)                                         # (B, T, V)

        loss: Optional[torch.Tensor] = None

        # ==========================================
        # Case 1: training / teacher forcing
        # ==========================================
        if labels is not None:
            # For outputs, expose deterministic logits (base LM)
            logits = logits_base

            # Standard causal shift
            logits_shift = logits_base[:, :-1, :]                             # (B, T-1, V)
            targets = labels[:, 1:]                                           # (B, T-1)

            if attention_mask is not None:
                attn = attention_mask[:, 1:]
            else:
                attn = torch.ones_like(targets, dtype=torch.long)

            valid = (targets != self.ignore_index) & (attn > 0)               # (B, T-1)

            B, Tm1, V = logits_shift.size()

            if valid.any():
                # Flatten
                logits_flat = logits_shift.reshape(-1, V)                     # (B*(T-1), V)
                targets_flat = targets.reshape(-1)                            # (B*(T-1),)
                valid_flat = valid.reshape(-1)                                # (B*(T-1),)

                # CE on deterministic base logits
                # (ignore_index handled via masking)
                ce_all = F.cross_entropy(
                    logits_flat,
                    targets_flat.clamp_min(0),
                    reduction="none",
                )                                                             # (B*(T-1),)
                ce_valid = ce_all[valid_flat]                                 # (N_valid,)

                # s_t = log σ_t^2 = 2 * logsigma
                logsigma_shift = logsigma_full[:, :-1]                        # (B, T-1)
                s_all = (2.0 * logsigma_shift).reshape(-1)                    # (B*(T-1),)
                s_valid = s_all[valid_flat]                                   # (N_valid,)

                # Kendall-style heteroscedastic CE:
                # L_t = exp(-s_t) * CE_t + s_t
                loss_tokens = torch.exp(-s_valid) * ce_valid + s_valid
                loss = loss_tokens.mean()
            else:
                loss = logits_base.new_tensor(0.0)

        # ==========================================
        # Case 2: pure forward / generation
        # ==========================================
        else:
            if inject_noise_for_logits:
                eps = torch.randn_like(logits_base)
                logits = logits_base + sigma_full.unsqueeze(-1) * eps
            else:
                logits = logits_base

        out = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=getattr(base_out, "past_key_values", None),
            hidden_states=getattr(base_out, "hidden_states", None),
            attentions=getattr(base_out, "attentions", None),
            cross_attentions=getattr(base_out, "cross_attentions", None),
        )

        # Aux fields for analysis/visualisation
        out.aux = {
            "logsigma": logsigma_full,     # log σ_t
            "sigma": sigma_full,           # σ_t
            "logits_base": logits_base,    # deterministic logits
        }
        return out

    # -----------------------------
    # Convenience utils
    # -----------------------------
    @torch.no_grad()
    def get_sigma_map(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Return per-token σ_t for a batch without computing loss.
        Shape: (B, T).
        """
        h, _ = self._forward_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        logsigma_raw = self.fc_logsigma(h).squeeze(-1)
        _, sigma = self._sigma_from_logsigma(logsigma_raw)
        return sigma