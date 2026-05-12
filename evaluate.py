import torch
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

@hydra.main(version_base="1.3", config_path=".", config_name="eval_config")
def evaluate(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # 1. Setup Dataloader (The Ground Truth Generator)
    normaliser = {
        "permeability": (cfg.normaliser.permeability.mean, cfg.normaliser.permeability.std_dev),
        "darcy": (cfg.normaliser.darcy.mean, cfg.normaliser.darcy.std_dev),
    }
    
    dataloader = Darcy2D(
        resolution=cfg.eval.resolution,
        batch_size=cfg.eval.batch_size,
        normaliser=normaliser,
    )

    # 2. Load the Model based on YAML type
    if cfg.model_type == "fno" or cfg.model_type == "pino":
        model = FNO(
            in_channels=1,
            out_channels=1,
            num_fno_modes=cfg.arch.fno_modes,
            num_fno_layers=cfg.arch.fno_layers,
            latent_channels=cfg.arch.latent_channels,
            dimension=2,
            padding=9
        )
    elif cfg.model_type == "pinn":
        model = FullyConnected(
            in_features=3, # x, y, a
            out_features=1,
            num_layers=cfg.arch.layers,
            layer_size=cfg.arch.layer_size,
            activation_fn="silu"
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    model.to(device)
    
    # Load weights
    print(f"Loading weights from {cfg.ckpt_path}")
    model.load_state_dict(torch.load(cfg.ckpt_path, map_user=device))
    model.eval()

    # 3. Evaluation Loop
    total_mse = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= cfg.eval.num_batches: break
            
            perm = batch["permeability"].to(device)
            target = batch["darcy"].to(device)

            if cfg.model_type in ["fno", "pino"]:
                # FNO/PINO expects the whole image [B, 1, H, W]
                prediction = model(perm)
            else:
                # PINN expects coordinates. We must build a grid for evaluation.
                B, _, H, W = perm.shape
                x = torch.linspace(0, 1, H, device=device)
                y = torch.linspace(0, 1, W, device=device)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                
                # Reshape to [B*H*W, 1] for point-wise inference
                coords = torch.stack([grid_x, grid_y], dim=-1).repeat(B, 1, 1, 1).view(-1, 2)
                a_vals = perm.permute(0, 2, 3, 1).reshape(-1, 1)
                
                input_pinn = torch.cat([coords, a_vals], dim=-1)
                prediction = model(input_pinn).view(B, 1, H, W)

            # Calculate Error
            error = torch.mean((prediction - target)**2)
            total_mse += error.item()

    avg_mse = total_mse / cfg.eval.num_batches
    print(f"--- Evaluation Results ---")
    print(f"Model: {cfg.model_type.upper()}")
    print(f"Resolution: {cfg.eval.resolution}x{cfg.eval.resolution}")
    print(f"Average MSE: {avg_mse:.6e}")

if __name__ == "__main__":
    evaluate()