from pathlib import Path

import torch

from rlbench.datasets.imitation_learning.cached_dataset import get_temporal_cached_dataloader

from src.models.platonic_transformer.groups import PLATONIC_GROUPS
from src.policies.platonic_policy import (
    PlatonicDiffusionPolicy,
    PlatonicDiffusionPolicyConfig,
    _features_to_pose,
    _pack_features,
    _pose_to_features,
    _unpack_features,
)
from src.utils.geometry import quaternion_to_matrix


CACHE_ROOT = Path("/home/dknigge/project_dir/data/robotics/rlbench/imitation_learning")


def _rotate_vectors(vectors: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate the vector features (B, T, 3, 3) by a shared 3x3 rotation."""
    return torch.einsum("ij,...kj->...ki", rotation, vectors)


def _translate_vectors(vectors: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Translate the position channel (index 0) of vector features."""
    translated = vectors.clone()
    translated[..., 0, :] = translated[..., 0, :] + translation.to(
        dtype=vectors.dtype, device=vectors.device
    )
    return translated


def _rotate_pose_sequence(pose: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate pose tensors by first converting to the scalar/vector format."""
    scalars, vectors = _pose_to_features(pose)
    rotated_vectors = _rotate_vectors(vectors, rotation)
    return _features_to_pose(scalars, rotated_vectors)


def _translate_pose_sequence(pose: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Translate pose tensors by shifting position channels."""
    scalars, vectors = _pose_to_features(pose)
    translated_vectors = _translate_vectors(vectors, translation)
    return _features_to_pose(scalars, translated_vectors)


def _rototranslate_pose_sequence(
    pose: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    """Apply rotation followed by translation to pose tensors."""
    return _translate_pose_sequence(_rotate_pose_sequence(pose, rotation), translation)


def _rotate_packed(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate packed (flattened) action tensors used by the diffusion scheduler."""
    scalars, vectors = _unpack_features(tensor, vector_channels=3, scalar_channels=1)
    rotated_vectors = _rotate_vectors(vectors, rotation)
    return _pack_features(scalars, rotated_vectors)


def _translate_packed(tensor: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Translate the position channel within packed action tensors."""
    scalars, vectors = _unpack_features(tensor, vector_channels=3, scalar_channels=1)
    translated_vectors = _translate_vectors(vectors, translation)
    return _pack_features(scalars, translated_vectors)


def _rototranslate_packed(
    tensor: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    """Apply rotation followed by translation within packed action tensors."""
    return _translate_packed(_rotate_packed(tensor, rotation), translation)


def _fetch_batch():
    """Load a single sample so we can test equivariance with real data."""
    dataloader = get_temporal_cached_dataloader(
        CACHE_ROOT,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    return batch["observation"]["proprio_sequence"], batch["action"]


def _random_quaternion(batch_shape: tuple[int, ...]) -> torch.Tensor:
    q = torch.randn(*batch_shape, 4)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def test_platonic_policy_equivariance():
    """Noise predictions should rotate exactly with inputs."""
    observations, actions = _fetch_batch()
    # observations shape: (B=1, T_obs, 8); actions shape: (B=1, T_action, 8)
    group = PLATONIC_GROUPS["icosahedron"]

    cfg = PlatonicDiffusionPolicyConfig(
        context_length=observations.shape[1],
        horizon=actions.shape[1],
        hidden_dim=group.G * 2,
        num_layers=1,
        num_heads=group.G,
        solid_name="icosahedron",
        ffn_dim_factor=2,
        noise_scheduler_kwargs={
            "num_train_timesteps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
        num_inference_steps=10,
    )
    policy = PlatonicDiffusionPolicy(cfg).cpu()
    policy.eval()

    obs_scalars, obs_vectors = policy._split_observations(observations)
    action_scalars, action_vectors = policy._split_actions(actions)
    action_model = _pack_features(action_scalars, action_vectors)  # (B, T_action, 10)

    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(
        action_model.shape,
        generator=generator,
        device=action_model.device,
        dtype=action_model.dtype,
    )
    timesteps = torch.randint(
        0,
        policy.scheduler.config.num_train_timesteps,
        (actions.shape[0],),
        generator=generator,
        dtype=torch.long,
    )
    noisy = policy.scheduler.add_noise(action_model, noise, timesteps)  # forward diffusion
    noisy_scalars, noisy_vectors = _unpack_features(
        noisy, vector_channels=3, scalar_channels=1
    )
    pred_noise = policy._tokenise(
        obs_scalars, obs_vectors, noisy_scalars, noisy_vectors, timesteps
    )

    rotation = group.elements[5].to(dtype=observations.dtype)
    observations_rot = _rotate_pose_sequence(observations, rotation)
    actions_rot = _rotate_pose_sequence(actions, rotation)

    obs_rot_scalars, obs_rot_vectors = policy._split_observations(observations_rot)
    action_rot_scalars, action_rot_vectors = policy._split_actions(actions_rot)
    action_rot_model = _pack_features(action_rot_scalars, action_rot_vectors)

    noise_rot = _rotate_packed(noise, rotation)
    noisy_rot = policy.scheduler.add_noise(action_rot_model, noise_rot, timesteps)
    assert torch.allclose(noisy_rot, _rotate_packed(noisy, rotation), atol=1e-5)

    noisy_rot_scalars, noisy_rot_vectors = _unpack_features(
        noisy_rot, vector_channels=3, scalar_channels=1
    )
    pred_noise_rot = policy._tokenise(
        obs_rot_scalars,
        obs_rot_vectors,
        noisy_rot_scalars,
        noisy_rot_vectors,
        timesteps,
    )
    assert torch.allclose(pred_noise_rot, _rotate_packed(pred_noise, rotation), atol=2e-4)


def test_rotate_packed_consistency():
    group = PLATONIC_GROUPS["icosahedron"]
    tensor = torch.randn(2, 5, 10)
    rotation = group.elements[3].to(dtype=tensor.dtype)

    rotated = _rotate_packed(tensor, rotation)
    scalars, vectors = _unpack_features(rotated, vector_channels=3, scalar_channels=1)
    original_scalars, original_vectors = _unpack_features(
        tensor, vector_channels=3, scalar_channels=1
    )
    manual_vectors = torch.einsum("ij,...kj->...ki", rotation, original_vectors)

    assert torch.allclose(scalars, original_scalars)
    assert torch.allclose(vectors, manual_vectors)


def test_tokenise_equivariance_random_actions():
    observations, _ = _fetch_batch()
    group = PLATONIC_GROUPS["icosahedron"]

    cfg = PlatonicDiffusionPolicyConfig(
        context_length=observations.shape[1],
        horizon=4,
        hidden_dim=group.G * 2,
        num_layers=1,
        num_heads=group.G,
        solid_name="icosahedron",
        ffn_dim_factor=2,
        noise_scheduler_kwargs={
            "num_train_timesteps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
        num_inference_steps=10,
    )
    policy = PlatonicDiffusionPolicy(cfg).cpu()
    policy.eval()

    observations = observations
    rotation = group.elements[11].to(dtype=observations.dtype)
    observations_rot = _rotate_pose_sequence(observations, rotation)

    rand_pos = torch.randn(observations.shape[0], cfg.horizon, 3)
    rand_quat = _random_quaternion((observations.shape[0], cfg.horizon))
    rand_grip = torch.randn(observations.shape[0], cfg.horizon, 1)
    random_actions = torch.cat([rand_pos, rand_quat, rand_grip], dim=-1)
    random_actions_rot = _rotate_pose_sequence(random_actions, rotation)

    obs_scalars, obs_vectors = policy._split_observations(observations)
    action_scalars, action_vectors = policy._split_actions(random_actions)
    action_model = _pack_features(action_scalars, action_vectors)

    obs_rot_scalars, obs_rot_vectors = policy._split_observations(observations_rot)
    action_rot_scalars, action_rot_vectors = policy._split_actions(random_actions_rot)
    action_rot_model = _pack_features(action_rot_scalars, action_rot_vectors)

    timesteps = torch.randint(0, policy.scheduler.config.num_train_timesteps, (observations.shape[0],))
    noise = torch.randn_like(action_model)
    noisy = policy.scheduler.add_noise(action_model, noise, timesteps)
    noisy_scalars, noisy_vectors = _unpack_features(noisy, vector_channels=3, scalar_channels=1)
    pred_noise = policy._tokenise(
        obs_scalars, obs_vectors, noisy_scalars, noisy_vectors, timesteps
    )

    noisy_rot = policy.scheduler.add_noise(action_rot_model, _rotate_packed(noise, rotation), timesteps)
    noisy_rot_scalars, noisy_rot_vectors = _unpack_features(noisy_rot, vector_channels=3, scalar_channels=1)
    pred_noise_rot = policy._tokenise(
        obs_rot_scalars,
        obs_rot_vectors,
        noisy_rot_scalars,
        noisy_rot_vectors,
        timesteps,
    )

    assert torch.allclose(pred_noise_rot, _rotate_packed(pred_noise, rotation), atol=2e-4)

def test_sampling_equivariance_deterministic():
    observations, _ = _fetch_batch()
    group = PLATONIC_GROUPS["icosahedron"]

    cfg = PlatonicDiffusionPolicyConfig(
        context_length=observations.shape[1],
        horizon=4,
        hidden_dim=group.G * 2,
        num_layers=1,
        num_heads=group.G,
        solid_name="icosahedron",
        ffn_dim_factor=2,
        noise_scheduler_kwargs={
            "num_train_timesteps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
        num_inference_steps=10,
    )
    policy = PlatonicDiffusionPolicy(cfg).cpu()
    policy.eval()

    noise = torch.randn(
        (
            observations.shape[0],
            cfg.horizon,
            cfg.vector_channels * 3 + cfg.scalar_channels,
        )
    )
    scalars, vectors = policy.sample_actions(
        observations,
        initial_noise=noise,
        deterministic=True,
        return_features=True,
    )
    actions = _features_to_pose(scalars, vectors).cpu()

    rotation = group.elements[7].to(dtype=observations.dtype)
    observations_rot = _rotate_pose_sequence(observations, rotation)

    noise_rot = _rotate_packed(noise, rotation)
    scalars_rot, vectors_rot = policy.sample_actions(
        observations_rot,
        initial_noise=noise_rot,
        deterministic=True,
        return_features=True,
    )
    rot_vectors = torch.einsum("ij,...kj->...ki", rotation, vectors)
    assert torch.allclose(scalars_rot.cpu(), scalars.cpu(), atol=1e-3)
    assert torch.allclose(vectors_rot.cpu(), rot_vectors.cpu(), atol=5e-4)

    actions_rot = _features_to_pose(scalars_rot, vectors_rot).cpu()

    rotated_actions = _rotate_pose_sequence(actions, rotation)

    def canonicalise(pose: torch.Tensor) -> torch.Tensor:
        quat = pose[..., 3:7]
        sign = torch.where(quat[..., -1:] < 0, -1.0, 1.0)
        pose = pose.clone()
        pose[..., 3:7] = quat * sign
        return pose

    actions_rot = canonicalise(actions_rot)
    rotated_actions = canonicalise(rotated_actions)

    assert torch.allclose(actions_rot[..., :3], rotated_actions[..., :3], atol=5e-4)
    assert torch.allclose(actions_rot[..., 7:], rotated_actions[..., 7:], atol=5e-4)

    rot_m = quaternion_to_matrix(actions_rot[..., 3:7])
    ref_m = quaternion_to_matrix(rotated_actions[..., 3:7])
    assert torch.allclose(rot_m, ref_m, atol=5e-4)


def test_sampling_equivariance_rototranslation():
    observations, _ = _fetch_batch()
    group = PLATONIC_GROUPS["icosahedron"]

    cfg = PlatonicDiffusionPolicyConfig(
        context_length=observations.shape[1],
        horizon=4,
        hidden_dim=group.G * 2,
        num_layers=1,
        num_heads=group.G,
        solid_name="icosahedron",
        ffn_dim_factor=2,
        noise_scheduler_kwargs={
            "num_train_timesteps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
        num_inference_steps=10,
    )
    policy = PlatonicDiffusionPolicy(cfg).cpu()
    policy.eval()

    noise = torch.randn(
        (
            observations.shape[0],
            cfg.horizon,
            cfg.vector_channels * 3 + cfg.scalar_channels,
        )
    )
    scalars, vectors = policy.sample_actions(
        observations,
        initial_noise=noise,
        deterministic=True,
        return_features=True,
    )
    actions = _features_to_pose(scalars, vectors).cpu()

    rotation = group.elements[9].to(dtype=observations.dtype)
    translation = torch.tensor([0.15, -0.08, 0.05], dtype=observations.dtype)

    observations_rt = _rototranslate_pose_sequence(observations, rotation, translation)

    noise_rt = _rototranslate_packed(noise, rotation, translation)
    scalars_rt, vectors_rt = policy.sample_actions(
        observations_rt,
        initial_noise=noise_rt,
        deterministic=True,
        return_features=True,
    )
    actions_rt = _features_to_pose(scalars_rt, vectors_rt).cpu()

    reference = _rototranslate_pose_sequence(actions, rotation, translation)

    def canonicalise(pose: torch.Tensor) -> torch.Tensor:
        quat = pose[..., 3:7]
        sign = torch.where(quat[..., -1:] < 0, -1.0, 1.0)
        pose = pose.clone()
        pose[..., 3:7] = quat * sign
        return pose

    actions_rt = canonicalise(actions_rt)
    reference = canonicalise(reference)

    assert torch.allclose(actions_rt[..., :3], reference[..., :3], atol=5e-4)
    assert torch.allclose(actions_rt[..., 7:], reference[..., 7:], atol=5e-4)
    rot_m = quaternion_to_matrix(actions_rt[..., 3:7])
    ref_m = quaternion_to_matrix(reference[..., 3:7])
    assert torch.allclose(rot_m, ref_m, atol=5e-4)
