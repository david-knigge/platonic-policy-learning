"""Minimal diffusion policy built around the DiT encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from src.models.dit import DiffusionTransformer, DiffusionTransformerConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        half_dim = max(1, dim // 2)
        inv_freq = base ** (-torch.arange(half_dim, dtype=torch.float32) / max(1, half_dim))  # (half_dim,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 1:
            indices = indices.unsqueeze(1)  # (B, 1)
        values = indices.to(self.inv_freq.dtype)  # (B, N)
        angles = values.unsqueeze(-1) * self.inv_freq  # (B, N, half_dim)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, N, min(dim, 2*half_dim))
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(*emb.shape[:-1], self.dim - emb.shape[-1], device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=-1)  # (B, N, dim)
        return emb


@dataclass
class DiTDiffusionPolicyConfig:
    # Number of observation tokens the policy attends over before action horizon.
    context_length: int
    # Length of the action rollout the policy predicts per sample.
    horizon: int
    # Dimensionality of each observation vector.
    obs_dim: int
    # Dimensionality of each action vector.
    action_dim: int
    # Hidden size shared across transformer, embeddings, and output head.
    hidden_dim: int
    # Number of transformer blocks stacked in the DiT backbone.
    num_layers: int
    # Count of attention heads per block.
    num_heads: int
    # Width of the MLP sub-layer inside each transformer block.
    mlp_dim: int
    # Dropout controlling residual and feedforward stochasticity.
    dropout: float = 0.0
    # Dropout specific to attention probability matrices.
    attention_dropout: float = 0.0
    # Activation function used by the MLP branch.
    activation: str = "gelu"
    # Whether LayerNorm precedes the attention/MLP (AdaLN-Zero prefers True).
    norm_first: bool = True
    # Numerical epsilon fed into LayerNorm for stability.
    layer_norm_eps: float = 1e-5
    # Hidden dimension for simple linear embeddings of scalar signals.
    scalar_embedding_hidden_dim: int = 128
    # Base frequency for the absolute time positional encoding.
    time_embedding_base: float = 10000.0
    # Base frequency for the diffusion timestep encoding.
    diffusion_embedding_base: float = 10000.0
    # Number of sampling steps retained for later inference usage.
    num_inference_steps: int = 50
    # Parameters forwarded into the diffusers DDPM scheduler.
    noise_scheduler_kwargs: Dict[str, object] | None = None


class DiTDiffusionPolicy(nn.Module):
    def __init__(self, cfg: DiTDiffusionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Translate policy hyper-parameters into a DiT backbone configuration.
        transformer_cfg = DiffusionTransformerConfig(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation=cfg.activation,
            norm_first=cfg.norm_first,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.transformer = DiffusionTransformer(transformer_cfg)

        # Linear projections turn observations and actions into token embeddings.
        self.obs_encoder = nn.Linear(cfg.obs_dim, cfg.hidden_dim)
        self.action_encoder = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

        # Positional encoders tag each token with absolute time information.
        self.world_time_embedder = SinusoidalTimeEmbedding(cfg.hidden_dim, base=cfg.time_embedding_base)
        self.diffusion_time_embedder = SinusoidalTimeEmbedding(
            cfg.hidden_dim, base=cfg.diffusion_embedding_base
        )
        self.diffusion_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        # Learned embeddings differentiate between observation and action token types.
        node_types = torch.randn(2, cfg.hidden_dim) * 0.02  # (2, D)
        self.node_type_embeddings = nn.Parameter(node_types)

        # Instantiate a DDPM scheduler with user-provided hyperparameters.
        scheduler_kwargs = dict(cfg.noise_scheduler_kwargs or {})
        self.scheduler = DDPMScheduler(**scheduler_kwargs)
        self.num_inference_steps = cfg.num_inference_steps

        # Pre-compute absolute time indices for observations and actions.
        context_idx = torch.arange(cfg.context_length, dtype=torch.float32).unsqueeze(0)  # (1, To)
        self.register_buffer("context_time_indices", context_idx, persistent=False)
        action_idx = torch.arange(cfg.horizon, dtype=torch.float32).unsqueeze(0)  # (1, Ta)
        self.register_buffer("action_time_indices", action_idx, persistent=False)

    # ------------------------------------------------------------------
    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        # Map each observation vector into the common hidden space.
        tokens = self.obs_encoder(observations)  # (B, To, D)
        batch = tokens.shape[0]
        times = self.context_time_indices.expand(batch, -1)  # (B, To)
        time_emb = self.world_time_embedder(times)  # (B, To, D)
        tokens = tokens + time_emb  # (B, To, D)
        tokens = tokens + self.node_type_embeddings[0].view(1, 1, -1)  # (B, To, D)
        return tokens

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # Embed the (possibly noisy) action tokens using the same hidden dimensionality.
        tokens = self.action_encoder(actions)  # (B, Ta, D)
        batch = tokens.shape[0]
        times = self.action_time_indices.expand(batch, -1) + float(self.cfg.context_length)  # (B, Ta)
        time_emb = self.world_time_embedder(times)  # (B, Ta, D)
        tokens = tokens + time_emb  # (B, Ta, D)
        tokens = tokens + self.node_type_embeddings[1].view(1, 1, -1)  # (B, Ta, D)
        return tokens

    def _diffusion_condition(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Encode the diffusion step and project it so AdaLN-Zero can modulate residual streams.
        emb = self.diffusion_time_embedder(timesteps.float().unsqueeze(1))[:, 0, :]  # (B, D)
        return self.diffusion_proj(emb)  # (B, D)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Pull raw observations and actions from the dataloader mini-batch.
        observations = batch["observations"]
        actions = batch["actions"]

        # Sample standard Gaussian noise for the forward diffusion process.
        noise = torch.randn_like(actions)  # (B, Ta, action_dim)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
            dtype=torch.long,
        )  # (B,)
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)  # (B, Ta, action_dim)

        # Convert clean observations and perturbed actions into joint token sequence.
        obs_tokens = self._encode_observations(observations)  # (B, To, D)
        action_tokens = self._encode_actions(noisy_actions)  # (B, Ta, D)
        tokens = torch.cat([obs_tokens, action_tokens], dim=1)  # (B, To+Ta, D)

        # Condition the DiT backbone on the diffusion timestep embedding.
        diffusion_cond = self._diffusion_condition(timesteps)  # (B, D)
        encoded = self.transformer(tokens, diffusion_time_cond=diffusion_cond)  # (B, To+Ta, D)

        # Take the tail slice corresponding to action tokens and predict the de-noised residual.
        pred = self.output_head(encoded[:, -self.cfg.horizon :, :])  # (B, Ta, action_dim)

        # Compare the network prediction to the original Gaussian noise; return scalars for logging.
        loss = F.mse_loss(pred, noise)  # scalar
        metrics = {"mse": float(loss.detach().cpu())}
        return loss, metrics

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.compute_loss(batch)

    def sample_actions(
        self, observations: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        # Infer batch sizing so the sampler emits a matching action rollout.
        batch_size = observations.shape[0]  # ()
        device = observations.device  # torch.device

        # Prepare a Gaussian starting point for the reverse diffusion trajectory.
        sample = torch.randn(
            (batch_size, self.cfg.horizon, self.cfg.action_dim),
            generator=generator,
            device=device,
        )  # (B, Ta, action_dim)

        # Encode observations once since they stay fixed throughout denoising.
        obs_tokens = self._encode_observations(observations)  # (B, To, D)

        # Configure the scheduler timesteps for inference.
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)  # (num_steps,)

        # Replay the reverse diffusion process until we reach the clean action manifold.
        for timestep in self.scheduler.timesteps:  # (num_steps,)
            # Broadcast the scalar timestep across the batch for conditioning.
            timesteps = torch.full(
                (batch_size,),
                timestep,
                device=device,
                dtype=torch.long,
            )  # (B,)

            # Encode the current noisy action tokens under the DiT backbone space.
            action_tokens = self._encode_actions(sample)  # (B, Ta, D)

            # Concatenate observation and action tokens for joint attention.
            tokens = torch.cat([obs_tokens, action_tokens], dim=1)  # (B, To+Ta, D)

            # Generate the adaptive LayerNorm modulation from the diffusion timestep.
            diffusion_cond = self._diffusion_condition(timesteps)  # (B, D)

            # Run the DiT transformer to predict the score (noise) at this timestep.
            encoded = self.transformer(
                tokens,
                diffusion_time_cond=diffusion_cond,
            )  # (B, To+Ta, D)

            # Project the action slice back into the action space to recover noise estimates.
            noise_pred = self.output_head(encoded[:, -self.cfg.horizon :, :])  # (B, Ta, action_dim)

            # Apply the scheduler update to obtain the next, slightly cleaner action sample.
            scheduler_output = self.scheduler.step(
                noise_pred,
                timestep,
                sample,
                generator=generator,
            )  # prev_sample: (B, Ta, action_dim)

            sample = scheduler_output.prev_sample  # (B, Ta, action_dim)

        # Return the denoised actions matching the dataset's action tensor layout.
        return sample  # (B, Ta, action_dim)


__all__ = ["DiTDiffusionPolicy", "DiTDiffusionPolicyConfig"]
