import torch

from src.policies.platonic_policy import (
    _features_to_pose,
    _pack_features,
    _pose_to_features,
    _unpack_features,
)
from src.utils.geometry import matrix_to_quaternion, quaternion_to_matrix


def random_quaternion(batch_shape: tuple[int, ...] = ()) -> torch.Tensor:
    """Sample random unit quaternions for the requested batch shape."""
    q = torch.randn(*batch_shape, 4)
    return q / q.norm(dim=-1, keepdim=True)


def test_quaternion_matrix_round_trip():
    """Ensure quaternion -> matrix -> quaternion returns the same orientation."""
    q = random_quaternion((16,))
    matrix = quaternion_to_matrix(q)  # (16, 3, 3)
    q_round = matrix_to_quaternion(matrix)  # (16, 4)
    canonical = torch.where(q[..., -1:] < 0, -q, q)  # enforce qw >= 0
    diff_direct = (canonical - q_round).abs().max()
    diff_flipped = (canonical + q_round).abs().max()
    assert min(diff_direct, diff_flipped) < 3e-5


def test_pose_feature_round_trip():
    """The pose <-> feature conversion used by the policy should be lossless."""
    position = torch.randn(2, 3, 3)
    quaternion = random_quaternion((2, 3))
    gripper = torch.rand(2, 3, 1)
    pose = torch.cat([position, quaternion, gripper], dim=-1)  # (2, 3, 8)

    scalars, vectors = _pose_to_features(pose)
    recovered = _features_to_pose(scalars, vectors)

    # Account for quaternion sign ambiguity before comparing.
    canonical = pose.clone()
    sign = torch.where(canonical[..., 6:7] < 0, -1.0, 1.0)
    canonical[..., 3:7] = canonical[..., 3:7] * sign

    assert torch.allclose(canonical[..., :3], recovered[..., :3], atol=1e-5)
    assert torch.allclose(canonical[..., 7:], recovered[..., 7:], atol=1e-5)

    orig_matrix = quaternion_to_matrix(canonical[..., 3:7])
    rec_matrix = quaternion_to_matrix(recovered[..., 3:7])
    assert torch.allclose(orig_matrix, rec_matrix, atol=1e-5)


def test_pack_unpack_symmetry():
    """Packing action features must be perfectly invertible."""
    scalars = torch.randn(4, 6, 1)
    vectors = torch.randn(4, 6, 3, 3)
    packed = _pack_features(scalars, vectors)
    unpack_scalars, unpack_vectors = _unpack_features(
        packed, vector_channels=3, scalar_channels=1
    )
    assert torch.allclose(scalars, unpack_scalars)
    assert torch.allclose(vectors, unpack_vectors)
