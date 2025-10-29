from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rlbench.datasets.imitation_learning.cached_dataset import get_temporal_cached_dataloader

from src.policies import DiTDiffusionPolicy, DiTDiffusionPolicyConfig


def parse_args() -> argparse.Namespace:
    # Configure the CLI so experiments can adjust hyperparameters without touching code.
    parser = argparse.ArgumentParser(description="Train a minimal DiT diffusion policy.")
    parser.add_argument("--cache-path", type=str, default="/home/dknigge/project_dir/data/robotics/rlbench/imitation_learning/", help="RLBench imitation learning cache root dir.")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="List of RLBench tasks to train on. Uses all if unspecified.")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory to store checkpoints.")

    # Training loop hyperparameters that control optimisation.
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker processes.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle RLBench samples each epoch.")
    parser.add_argument("--drop-last", action="store_true", help="Drop the final incomplete batch.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pinned-memory optimisation.")

    # Model architecture knobs exposed for quick sweeps.
    parser.add_argument("--hidden-dim", type=int, default=512, help="Transformer hidden size.")
    parser.add_argument("--num-layers", type=int, default=8, help="Transformer depth.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--mlp-dim", type=int, default=2048, help="Transformer MLP hidden size.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Token dropout.")
    parser.add_argument("--attention-dropout", type=float, default=0.0, help="Attention dropout.")

    # Diffusion noise schedule settings passed straight into the DDPM scheduler.
    parser.add_argument("--num-train-timesteps", type=int, default=1000, help="Noise scheduler train steps.")
    parser.add_argument("--beta-start", type=float, default=0.0001, help="Noise schedule beta start.")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Noise schedule beta end.")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear", help="Noise schedule type.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Sampling steps (for future use).")

    # Miscellaneous training conveniences.
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N epochs.")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping threshold (0 disables).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Decide whether to leverage GPU acceleration; fall back to CPU when unavailable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve rlbench dataloader from the cache root supplied on the CLI.
    cache_root = Path(args.cache_path).expanduser().resolve()
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache path not found: {cache_root}")
    
    # List all dirs under cache root as possible tasks, and list which are selected.
    available_tasks = [d.name for d in cache_root.iterdir() if d.is_dir()]
    if args.tasks is None:
        args.tasks = available_tasks
    else:
        for task in args.tasks:
            if task not in available_tasks:
                raise ValueError(f"Requested task '{task}' not found in cache at {cache_root}")
    print(f"Using tasks: {args.tasks} from available: {available_tasks}")

    dataloader = get_temporal_cached_dataloader(
        cache_root,
        tasks=args.tasks,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )

    # Peek at a reference batch so model dimensions align with cached dataset structure.
    sample = next(iter(dataloader))
    observations = sample["observation"]["proprio_sequence"]  # (B, To, obs_dim)
    actions = sample["action"]  # (B, Ta, action_dim)
    context_length = int(observations.shape[1])
    horizon = int(actions.shape[1])
    obs_dim = int(observations.shape[2])
    action_dim = int(actions.shape[2])

    scheduler_kwargs = {
        "num_train_timesteps": args.num_train_timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "beta_schedule": args.beta_schedule,
        "prediction_type": "epsilon",
    }

    # Construct the DiT-backed policy using inferred shapes plus CLI overrides.
    policy_cfg = DiTDiffusionPolicyConfig(
        context_length=context_length,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        noise_scheduler_kwargs=scheduler_kwargs,
        num_inference_steps=args.num_inference_steps,
    )
    policy = DiTDiffusionPolicy(policy_cfg).to(device)

    # AdamW optimiser keeps weight decay decoupled from gradient magnitude.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Ensure a place exists for checkpoints, even on the first run.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Policy architecture:\n")
    print(policy)
    print(
        f"\nParameters | total: {total_params:,} | trainable: {trainable_params:,} | device: {device.type}"
    )

    try:
        batches_per_epoch = len(dataloader)
    except TypeError:
        batches_per_epoch = None

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        policy.train()
        running_loss = 0.0
        batches_processed = 0

        progress = tqdm(
            dataloader,
            total=batches_per_epoch,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False,
        )

        for batch in progress:
            observations = batch["observation"]["proprio_sequence"].to(device)  # (B, To, obs_dim)
            actions = batch["action"].to(device)  # (B, Ta, action_dim)
            policy_batch = {
                "observations": observations,  # (B, To, obs_dim)
                "actions": actions,  # (B, Ta, action_dim)
            }

            # Standard diffusion training: zero gradients, compute loss, backprop, and step.
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy.compute_loss(policy_batch)  # loss: scalar
            loss.backward()

            if args.grad_clip > 0:
                # Optional gradient clipping for stability with large models or learning rates.
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)

            optimizer.step()
            batches_processed += 1
            running_loss += loss.item()
            avg_loss = running_loss / batches_processed
            progress.set_postfix(loss=f"{avg_loss:.6f}")

        progress.close()
        avg_loss = running_loss / max(1, batches_processed)
        # Report scalar loss so training progress is visible without extra tooling.
        tqdm.write(f"epoch {epoch:03d} | loss {avg_loss:.6f}")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            checkpoint_path = output_dir / f"policy_epoch{epoch:03d}.pt"
            # Persist model weights, optimiser state, and config for reproducibility.
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": policy.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(policy_cfg),
                },
                checkpoint_path,
            )
            print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
