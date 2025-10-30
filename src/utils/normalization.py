from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .stats import NormalizationStats


@dataclass
class NormalizationTransform:
    stats: NormalizationStats
    eps: float = 1e-6

    def _pos_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.stats.position_mean.to(device=tensor.device, dtype=tensor.dtype)

    def _pos_std(self, tensor: torch.Tensor) -> torch.Tensor:
        std = self.stats.position_std.to(device=tensor.device, dtype=tensor.dtype)
        return std.clamp_min(self.eps)

    def _col_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.stats.color_mean.to(device=tensor.device, dtype=tensor.dtype)

    def _col_std(self, tensor: torch.Tensor) -> torch.Tensor:
        std = self.stats.color_std.to(device=tensor.device, dtype=tensor.dtype)
        return std.clamp_min(self.eps)

    def normalize_positions(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self._pos_mean(tensor)) / self._pos_std(tensor)

    def denormalize_positions(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self._pos_std(tensor) + self._pos_mean(tensor)

    def normalize_colors(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self._col_mean(tensor)) / self._col_std(tensor)

    def denormalize_colors(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self._col_std(tensor) + self._col_mean(tensor)

    def normalize_action_positions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.clone()
        actions[..., :3] = self.normalize_positions(actions[..., :3])
        return actions

    def denormalize_action_positions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.clone()
        actions[..., :3] = self.denormalize_positions(actions[..., :3])
        return actions

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        proprio = proprio.clone()
        proprio[..., :3] = self.normalize_positions(proprio[..., :3])
        return proprio

    def denormalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        proprio = proprio.clone()
        proprio[..., :3] = self.denormalize_positions(proprio[..., :3])
        return proprio

    def normalize_point_cloud_sequence(self, sequence: List[List[dict]]) -> List[List[dict]]:
        normalized = []
        for views in sequence:
            view_list = []
            for view in views:
                points = view["points"].to(dtype=torch.float32)
                colors = view["colors"].to(dtype=torch.float32)
                view_list.append(
                    {
                        **view,
                        "points": self.normalize_positions(points),
                        "colors": self.normalize_colors(colors),
                    }
                )
            normalized.append(view_list)
        return normalized

    def denormalize_point_cloud_sequence(self, sequence: List[List[dict]]) -> List[List[dict]]:
        restored = []
        for views in sequence:
            view_list = []
            for view in views:
                points = view["points"].to(dtype=torch.float32)
                colors = view["colors"].to(dtype=torch.float32)
                view_list.append(
                    {
                        **view,
                        "points": self.denormalize_positions(points),
                        "colors": self.denormalize_colors(colors),
                    }
                )
            restored.append(view_list)
        return restored

    def normalize_batch(self, batch: dict) -> dict:
        obs = batch["observation"]
        normalized_pc = self.normalize_point_cloud_sequence(obs["point_cloud_sequence"])
        normalized_proprio = self.normalize_proprio(obs["proprio_sequence"])
        normalized_actions = self.normalize_action_positions(batch["action"])
        return {
            "observation": {
                **obs,
                "point_cloud_sequence": normalized_pc,
                "proprio_sequence": normalized_proprio,
            },
            "action": normalized_actions,
            **{k: v for k, v in batch.items() if k not in {"observation", "action"}},
        }
