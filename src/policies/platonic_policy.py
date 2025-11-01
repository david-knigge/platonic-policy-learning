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
def _pose_to_components(
    pose: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose pose tensors into scalar, orientation-vector, and position slots.

    - Scalars: gripper open/close values.
    - Orientation vectors: forward/up axes spanning the gripper frame.
    - Positions: end-effector positions in Cartesian space.
    """
    position = pose[..., :3]  # (..., 3)
    quaternion = pose[..., 3:7]  # (..., 4)
    grasp = pose[..., 7:8]  # (..., 1)
    forward, _, up = quaternion_to_frame(quaternion)  # each (..., 3)
    orientation = torch.stack([forward, up], dim=-2)  # (..., 2, 3)
    return grasp, orientation, position


def _components_to_pose(
    grasp: torch.Tensor,
    orientation: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct pose tensors from scalar, orientation, and position slots."""
    forward = orientation[..., 0, :]  # (..., 3)
    up = orientation[..., 1, :]  # (..., 3)
    quaternion = frame_to_quaternion(forward, up)  # (..., 4)
    return torch.cat([position, quaternion, grasp], dim=-1)  # (..., 8)


def _pack_components(
    grasp: torch.Tensor,
    orientation: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    """Flatten orientation vectors and concatenate positions + scalars."""
    orientation_flat = orientation.reshape(*orientation.shape[:-2], -1)  # (..., 6)
    return torch.cat([orientation_flat, position, grasp], dim=-1)  # (..., 10)


def _unpack_components(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`_pack_components`."""
    orientation_flat = tensor[..., :6]
    position = tensor[..., 6:9]
    grasp = tensor[..., 9:]
    orientation = orientation_flat.view(*tensor.shape[:-1], 2, 3)
    return grasp, orientation, position


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
    scalar_channels: int = 4
    # Number of vector-valued feature channels supplied on input.
    input_vector_channels: int = 2
    # Number of vector-valued channels predicted by the transformer head.
    output_vector_channels: int = 3
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
            input_dim_vec=cfg.input_vector_channels,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.scalar_channels,
            output_dim_vec=cfg.output_vector_channels,
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

        # Separate scalar embedders for proprio, actions, and per-point colours.
        self.proprio_scalar_encoder = nn.Sequential(
            nn.Linear(1, cfg.scalar_channels),
            nn.SiLU(),
            nn.Linear(cfg.scalar_channels, cfg.scalar_channels),
        )
        self.action_scalar_encoder = nn.Sequential(
            nn.Linear(1, cfg.scalar_channels),
            nn.SiLU(),
            nn.Linear(cfg.scalar_channels, cfg.scalar_channels),
        )
        self.action_scalar_decoder = nn.Linear(cfg.scalar_channels, 1)
        self.point_rgb_encoder = nn.Sequential(
            nn.Linear(3, cfg.scalar_channels),
            nn.SiLU(),
            nn.Linear(cfg.scalar_channels, cfg.scalar_channels),
        )

        # Orientation occupies two vector channels; channel 0 remains reserved for positions.
        self.orientation_channels = self.cfg.input_vector_channels
        if self.cfg.output_vector_channels < self.orientation_channels + 1:
            raise ValueError("output_vector_channels must be at least orientation_channels + 1.")
        self.packed_action_dim = self.orientation_channels * 3 + 3 + 1  # (orientation, position, gripper)

    # ------------------------------------------------------------------
    # Utility helpers

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _tokenise(
        self,
        obs_scalars: torch.Tensor,
        obs_vectors: torch.Tensor,
        obs_positions: torch.Tensor,
        action_gripper: torch.Tensor,
        action_orientation: torch.Tensor,
        action_positions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Run the transformer and predict action-token noise."""
        # Embed action scalars so transformer sees rich scalar features.
        action_scalar_features = self.action_scalar_encoder(action_gripper)  # (B, N_action, scalar_channels)

        # Only orientation is provided through vector features; positions live in the `pos` argument.
        B, N_action = action_orientation.shape[:2]
        action_vector_features = torch.zeros(
            B,
            N_action,
            self.cfg.input_vector_channels,
            3,
            device=action_orientation.device,
            dtype=action_orientation.dtype,
        )
        action_vector_features[..., : self.orientation_channels, :] = action_orientation

        # Concatenate context tokens with the (noisy) action tokens.
        tokens_scalars = torch.cat(
            [obs_scalars, action_scalar_features],
            dim=1,
        )
        tokens_vectors = torch.cat(
            [obs_vectors, action_vector_features],
            dim=1,
        )
        token_positions = torch.cat([obs_positions, action_positions], dim=1)

        scalar_pred, vector_pred = self.transformer(
            scalars=tokens_scalars,
            pos=token_positions,
            vec=tokens_vectors,
            time_conditioning=timesteps,
        )

        # Transformer returns predictions for both context + horizon tokens.
        pred_scalar_features = scalar_pred[:, -self.cfg.horizon :, :]  # (B, N_action, scalar_channels)
        pred_vectors = vector_pred[
            :,
            -self.cfg.horizon :,
            :,
            :,
        ]  # (B, N_action, output_vector_channels, 3)
        pred_gripper = self.action_scalar_decoder(pred_scalar_features)  # (B, N_action, 1)
        pred_orientation = pred_vectors[..., 1:, :]  # (B, N_action, 2, 3)
        pred_positions = pred_vectors[..., 0, :]  # (B, N_action, 3)
        return _pack_components(pred_gripper, pred_orientation, pred_positions)

    def _build_context_tokens(
        self,
        proprio: torch.Tensor,
        point_cloud: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine proprio and point cloud observations into transformer tokens."""
        proprio_gripper, proprio_orientation, proprio_position = _pose_to_components(proprio)
        anchor = proprio_position[:, 0, :].clone()  # (B, 3)
        proprio_position = proprio_position - anchor[:, None, :]

        B, N_time = proprio.shape[:2]
        device = proprio.device
        dtype = proprio.dtype

        proprio_scalars = self.proprio_scalar_encoder(proprio_gripper)  # (B, N_time, scalar_channels)
        proprio_vectors = torch.zeros(
            B,
            N_time,
            self.cfg.input_vector_channels,
            3,
            device=device,
            dtype=dtype,
        )
        proprio_vectors[..., : self.orientation_channels, :] = proprio_orientation

        proprio_positions = proprio_position  # (B, N_time, 3)

        if point_cloud is None:
            return proprio_scalars, proprio_vectors, proprio_positions, anchor

        points = point_cloud["positions"].to(device=device, dtype=dtype)
        colors = point_cloud["colors"].to(device=device, dtype=dtype)
        _, _, N_points, _ = points.shape

        points = points - anchor[:, None, None, :]

        point_scalars = self.point_rgb_encoder(
            colors.reshape(B, N_time * N_points, 3)
        )  # (B, N_time * N_points, scalar_channels)
        point_vectors = torch.zeros(
            B,
            N_time * N_points,
            self.cfg.input_vector_channels,
            3,
            device=device,
            dtype=dtype,
        )
        point_positions = points.reshape(B, N_time * N_points, 3)

        combined_scalars = torch.cat([proprio_scalars, point_scalars], dim=1)
        combined_vectors = torch.cat([proprio_vectors, point_vectors], dim=1)
        combined_positions = torch.cat([proprio_positions, point_positions], dim=1)
        return combined_scalars, combined_vectors, combined_positions, anchor

    def _split_actions(
        self,
        actions: torch.Tensor,
        anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert action poses into scalar/vector slots centred at the anchor."""
        gripper, orientation, position = _pose_to_components(actions)
        position = position - anchor[:, None, :]
        return gripper, orientation, position

    # ------------------------------------------------------------------
    # Diffusion interface

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        proprio = batch["proprio"]
        point_cloud = batch.get("point_cloud")
        actions = batch["actions"]

        obs_scalars, obs_vectors, obs_positions, anchor = self._build_context_tokens(
            proprio,
            point_cloud,
        )
        action_gripper, action_orientation, action_positions = self._split_actions(actions, anchor)

        # Pack orientation, position, and scalar slots so the scheduler operates in a flat space.
        action_model = _pack_components(action_gripper, action_orientation, action_positions)
        noise = torch.randn_like(action_model)

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
        )  # (B, N_action, 10)
        noisy_gripper, noisy_orientation, noisy_positions = _unpack_components(
            noisy_actions,
        )

        # Predict the Gaussian noise residual under the Platonic transformer.
        pred_noise = self._tokenise(
            obs_scalars,
            obs_vectors,
            obs_positions,
            noisy_gripper,
            noisy_orientation,
            noisy_positions,
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
        proprio: torch.Tensor,
        point_cloud: Optional[Dict[str, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor:
        # Tokenise observations once; they stay fixed during ancestral sampling.
        obs_scalars, obs_vectors, obs_positions, anchor = self._build_context_tokens(
            proprio,
            point_cloud,
        )

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
                proprio.shape[0],
                self.cfg.horizon,
                self.packed_action_dim,
            )
            sample = torch.randn(
                sample_shape,
                generator=generator,
                device=self.device,
            )

        # Sweep the configured inference timesteps in reverse order.
        for timestep in self.scheduler.timesteps:
            # Convert packed action state into scalar/vector view for the model.
            noisy_gripper, noisy_orientation, noisy_positions = _unpack_components(
                sample,
            )
            timestep_tensor = torch.full(
                (proprio.shape[0],),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            # Platonic transformer predicts the denoising residual for horizon tokens.
            noise_pred = self._tokenise(
                obs_scalars,
                obs_vectors,
                obs_positions,
                noisy_gripper,
                noisy_orientation,
                noisy_positions,
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
            # Feed the next iterate back into the loop in packed form.
            sample = scheduler_step.prev_sample.to(dtype=proprio.dtype)

        # After finishing the reverse process, convert back into pose tensors.
        final_gripper, final_orientation, final_positions = _unpack_components(
            sample,
        )
        final_positions = final_positions + anchor[:, None, :]
        if return_features:
            return final_gripper, final_orientation, final_positions
        # Convert the final scalar/vector representation back into pose layout.
        actions = _components_to_pose(final_gripper, final_orientation, final_positions)
        return actions


__all__ = [
    "PlatonicDiffusionPolicy",
    "PlatonicDiffusionPolicyConfig",
    "_pose_to_components",
    "_components_to_pose",
    "_pack_components",
    "_unpack_components",
]
