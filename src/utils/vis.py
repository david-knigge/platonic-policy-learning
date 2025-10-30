from __future__ import annotations

import torch
import wandb


def point_cloud_with_actions(
    point_cloud_sequence,
    predicted_actions: torch.Tensor,
    gt_actions: torch.Tensor,
) -> wandb.Object3D:
    frame = point_cloud_sequence[-1]
    points = frame["points"].reshape(-1, 3).detach().cpu().to(torch.float32)
    colors = frame["colors"].reshape(-1, 3).detach().cpu().to(torch.float32)
    if colors.max() <= 1:
        colors = colors * 255.0
    colors = colors.clamp(0.0, 255.0)

    gt_positions = gt_actions[:, :3].detach().cpu().to(torch.float32)
    pred_positions = predicted_actions[:, :3].detach().cpu().to(torch.float32)

    gt_colors = torch.tensor([0.0, 255.0, 0.0], dtype=torch.float32).view(1, 3).repeat(gt_positions.shape[0], 1)
    pred_colors = torch.tensor([255.0, 0.0, 0.0], dtype=torch.float32).view(1, 3).repeat(pred_positions.shape[0], 1)

    merged_points = torch.cat([points, gt_positions, pred_positions], dim=0)
    merged_colors = torch.cat([colors, gt_colors, pred_colors], dim=0)

    cloud = torch.cat([merged_points, merged_colors], dim=1).numpy()
    return wandb.Object3D(cloud)
