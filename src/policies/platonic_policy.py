"""Diffusion policy built on top of the Platonic Transformer backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from src.models.platonic_transformer import DensePlatonicTransformer
from src.utils.geometry import (
    quaternion_to_frame,
    frame_to_quaternion,
)


# ---------------------------------------------------------------------------
def _pose_to_features(pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert pose tensors into a (scalar, vector) feature pair understood by the
    transformer.

    - Scalars: gripper open/close values.
    - Vectors: end-effector position, forward axis, and upward axis.
    """
    # pose shape: (..., 8) -> (x, y, z, qx, qy, qz, qw, grasp)
    position = pose[..., :3]  # (..., 3)
    quaternion = pose[..., 3:7]  # (..., 4)
    grasp = pose[..., 7:]  # (..., 1)
    forward, right, up = quaternion_to_frame(quaternion)  # each (..., 3)
    # We retain only forward/up for the vector features; right can be recovered.
    vectors = torch.stack(
        [position, forward, up],
        dim=-2,
    )  # (..., 3, 3)
    return grasp, vectors


def _features_to_pose(
    grasp: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct the original pose tensor from scalar/vector features.

    The first vector channel is treated as position; the remaining channels
    correspond to forward/up axes spanning the gripper frame.
    """
    position = vectors[..., 0, :]  # (..., 3)
    forward = vectors[..., 1, :]  # (..., 3)
    up = vectors[..., 2, :]  # (..., 3)
    quaternion = frame_to_quaternion(forward, up)  # (..., 4)
    return torch.cat([position, quaternion, grasp], dim=-1)  # (..., 8)


def _pack_features(grasp: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Flatten vector/scalar channels so the DDPM scheduler can inject noise."""
    # vectors shape: (B, T, 3 vectors, 3 dims) -> flatten vector axes.
    vector_flat = vectors.reshape(*vectors.shape[:-2], -1)  # (..., 9)
    return torch.cat([vector_flat, grasp], dim=-1)  # (..., 10)


def _unpack_features(
    tensor: torch.Tensor,
    *,
    vector_channels: int,
    scalar_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse operation of :func:`_pack_features`."""
    if scalar_channels == 0:
        grasp = tensor.new_zeros(*tensor.shape[:-1], 0)
        vector_flat = tensor
    else:
        grasp = tensor[..., -scalar_channels:]
        vector_flat = tensor[..., :-scalar_channels]
    vectors = vector_flat.view(*tensor.shape[:-1], vector_channels, 3)
    return grasp, vectors


# ---------------------------------------------------------------------------
# Policy configuration

@dataclass
class PlatonicDiffusionPolicyConfig:
    # Number of observation frames provided as context tokens.
    context_length: int
    # Action horizon (trajectory length) predicted by the policy.
    horizon: int
    # Transformer hidden size shared across scalar/vector streams.
    hidden_dim: int
    # Depth of the Platonic transformer (count of residual layers).
    num_layers: int
    # Number of attention heads per layer; often equals |G| for symmetry.
    num_heads: int
    # Platonic solid name selecting the equivariant symmetry group.
    solid_name: str
    # Expansion factor inside each feed-forward network branch.
    ffn_dim_factor: int = 4
    # Dropout applied to residual activations.
    dropout: float = 0.0
    # Drop-path rate for stochastic depth regularisation.
    drop_path_rate: float = 0.0
    # Average token outputs instead of taking the final action slice.
    mean_aggregation: bool = False
    # Switch between softmax and linear attention kernels.
    use_softmax_attention: bool = False
    # Scaling for rotary frequencies (RoPE) used by the backbone.
    rope_sigma: float = 1.0
    # Learnable frequency spectrum instead of fixed sinusoidal bands.
    learned_freqs: bool = True
    # Task-level routing for scalar features (e.g. dense vs sparse).
    scalar_task_level: str = "dense"
    # Task-level routing for vector features.
    vector_task_level: str = "dense"
    # Overrides forwarded to the diffusers DDIM scheduler.
    noise_scheduler_kwargs: Dict[str, object] | None = None
    # Number of reverse steps used at inference.
    num_inference_steps: int = 50

    # Count of scalar channels in packed action tokens.
    scalar_channels: int = 1
    # Count of vector channels (each a 3D feature).
    vector_channels: int = 3
    # Amount of DDIM stochasticity (0 = fully deterministic sampler).
    ddim_eta: float = 0.0


# ---------------------------------------------------------------------------
# Main policy

class PlatonicDiffusionPolicy(nn.Module):
    """Diffusion policy with Platonic rotational equivariance."""

    def __init__(self, cfg: PlatonicDiffusionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        scheduler_kwargs = dict(cfg.noise_scheduler_kwargs or {})
        scheduler_kwargs.setdefault("prediction_type", "epsilon")
        scheduler_kwargs.setdefault("clip_sample", False)
        # Shared diffusion scheduler for training and sampling.
        self.scheduler = DDIMScheduler(**scheduler_kwargs)
        self.num_inference_steps = cfg.num_inference_steps

        self.transformer = DensePlatonicTransformer(
            input_dim=cfg.scalar_channels,
            input_dim_vec=cfg.vector_channels,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.scalar_channels,
            output_dim_vec=cfg.vector_channels,
            nhead=cfg.num_heads,
            num_layers=cfg.num_layers,
            solid_name=cfg.solid_name,
            dropout=cfg.dropout,
            drop_path_rate=cfg.drop_path_rate,
            mean_aggregation=cfg.mean_aggregation,
            attention=cfg.use_softmax_attention,
            ffn_dim_factor=cfg.ffn_dim_factor,
            rope_sigma=cfg.rope_sigma,
            learned_freqs=cfg.learned_freqs,
            scalar_task_level=cfg.scalar_task_level,
            vector_task_level=cfg.vector_task_level,
            time_conditioning=True,
        )

    # ------------------------------------------------------------------
    # Utility helpers

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _tokenise(
        self,
        obs_scalars: torch.Tensor,
        obs_vectors: torch.Tensor,
        action_scalars: torch.Tensor,
        action_vectors: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Run the transformer and predict action-token noise."""
        # obs_scalars: (B, To, 1), action_scalars: (B, Ta, 1)
        tokens_scalars = torch.cat(
            [obs_scalars, action_scalars],
            dim=1,
        )  # (B, To + Ta, 1)
        # obs_vectors: (B, To, 3, 3), action_vectors: (B, Ta, 3, 3)
        tokens_vectors = torch.cat(
            [obs_vectors, action_vectors],
            dim=1,
        )  # (B, To + Ta, 3, 3)
        # Feed the position channel (index 0) as absolute coordinates.
        token_positions = tokens_vectors[..., 0, :]  # (B, To + Ta, 3)

        scalar_pred, vector_pred = self.transformer(
            scalars=tokens_scalars,
            pos=token_positions,
            vec=tokens_vectors,
            time_conditioning=timesteps,
        )

        # Transformer returns predictions for both context + horizon tokens.
        pred_scalars = scalar_pred[:, -self.cfg.horizon :, :]  # (B, Ta, 1)
        pred_vectors = vector_pred[
            :,
            -self.cfg.horizon :,
            :,
            :,
        ]  # (B, Ta, 3, 3)
        return _pack_features(pred_scalars, pred_vectors)

    def _split_observations(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert raw proprio tensors into scalar/vector channels."""
        grasp, vectors = _pose_to_features(
            observations
        )  # (B, To, 1), (B, To, 3, 3)
        return grasp, vectors

    def _split_actions(
        self,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrapper for readability – matches observation splitter."""
        return self._split_observations(actions)

    # ------------------------------------------------------------------
    # Diffusion interface

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch["observations"]
        actions = batch["actions"]

        # Convert tensors into the scalar/vector form used by the backbone.
        obs_scalars, obs_vectors = self._split_observations(
            observations
        )  # (B, To, 1/3/3)
        action_scalars, action_vectors = self._split_actions(
            actions
        )  # (B, Ta, 1/3/3)

        # Pack vectors + scalars so the scheduler operates in a flat space.
        action_model = _pack_features(
            action_scalars,
            action_vectors,
        )  # (B, Ta, 10)
        noise = torch.randn_like(action_model)  # Gaussian noise ~ (B, Ta, 10)

        # Sample diffusion timesteps and run forward diffusion.
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
            dtype=torch.long,
        )
        noisy_actions = self.scheduler.add_noise(
            action_model,
            noise,
            timesteps,
        )  # (B, Ta, 10)
        noisy_scalars, noisy_vectors = _unpack_features(
            noisy_actions,
            vector_channels=self.cfg.vector_channels,
            scalar_channels=self.cfg.scalar_channels,
        )

        # Predict the Gaussian noise residual under the Platonic transformer.
        pred_noise = self._tokenise(
            obs_scalars,
            obs_vectors,
            noisy_scalars,
            noisy_vectors,
            timesteps,
        )

        # Standard diffusion objective: match the sampled noise.
        loss = F.mse_loss(pred_noise, noise)
        metrics = {"mse": float(loss.detach().cpu())}
        return loss, metrics

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.compute_loss(batch)

    def sample_actions(
        self,
        observations: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor:
        # Tokenise observations once; they stay fixed during ancestral sampling.
        obs_scalars, obs_vectors = self._split_observations(
            observations
        )  # (B, To, 1/3/3)

        # Configure the reverse diffusion schedule on the current device.
        self.scheduler.set_timesteps(
            self.num_inference_steps,
            device=self.device,
        )

        # Start from isotropic Gaussian noise in the packed representation.
        if initial_noise is not None:
            sample = initial_noise.to(device=self.device)
        else:
            sample_shape = (
                observations.shape[0],
                self.cfg.horizon,
                self.cfg.vector_channels * 3 + self.cfg.scalar_channels,
            )
            sample = torch.randn(
                sample_shape,
                generator=generator,
                device=self.device,
            )

        # Sweep the configured inference timesteps in reverse order.
        for timestep in self.scheduler.timesteps:
            # Convert packed action state into scalar/vector view for the model.
            noisy_scalars, noisy_vectors = _unpack_features(
                sample,
                vector_channels=self.cfg.vector_channels,
                scalar_channels=self.cfg.scalar_channels,
            )
            timestep_tensor = torch.full(
                (observations.shape[0],),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            # Platonic transformer predicts the denoising residual for horizon tokens.
            noise_pred = self._tokenise(
                obs_scalars,
                obs_vectors,
                noisy_scalars,
                noisy_vectors,
                timestep_tensor,
            )

            # DDIM update: eta governs stochasticity, while eta=0 recovers the
            # purely deterministic ancestral trajectory.
            eta = 0.0 if deterministic else self.cfg.ddim_eta
            scheduler_step = self.scheduler.step(
                noise_pred,
                int(timestep),
                sample,
                eta=eta,
                generator=generator,
            )
            # Feed the next iterate back into the loop in packed (B, Ta, 10) form.
            sample = scheduler_step.prev_sample.to(dtype=observations.dtype)

        # After finishing the reverse process, convert back into pose tensors.
        final_scalars, final_vectors = _unpack_features(
            sample,
            vector_channels=self.cfg.vector_channels,
            scalar_channels=self.cfg.scalar_channels,
        )
        if return_features:
            return final_scalars, final_vectors
        # Convert the final scalar/vector representation back into pose layout.
        actions = _features_to_pose(final_scalars, final_vectors)
        return actions


__all__ = ["PlatonicDiffusionPolicy", "PlatonicDiffusionPolicyConfig"]
