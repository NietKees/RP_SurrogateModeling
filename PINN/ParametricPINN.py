import os
import sys
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from physicsnemo.utils import (
    StaticCaptureTraining,
    StaticCaptureEvaluateNoGrad,
    load_checkpoint,
    save_checkpoint,
)

# Assumed data/physics utilities based on your setup
from data_utils import get_darcy_setup
from physics_utils import get_physics_informer


# ============================================================
# P2INN Model Architecture
# ============================================================

class P2INN_Darcy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 1. Parameter Encoder (g_p): Outputs a dense latent feature map
        self.param_encoder = FNO(
            in_channels=1, 
            out_channels=128,           # Hardcoded latent dimension
            num_fno_modes=[12, 12],     # Hardcoded modes for 2D Darcy (64x64 resolution)
            num_fno_layers=4,           # Hardcoded FNO layers
            padding=9,                  # Hardcoded padding
            dimension=2,                # 2D problem
            latent_channels=128         # FNO width
        )
        
        # 2. Coordinate Encoder (Pure PyTorch MLP)
        # Replaces FullyConnected: 2 in_features -> 3 layers -> 64 out_features
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU()
        )
        
        # 3. Manifold Decoder Network (Pure PyTorch MLP)
        # Replaces FullyConnected: 128 (FNO) + 64 (Coord) = 192 in_features -> 4 layers -> 1 out_feature
        self.manifold_net = nn.Sequential(
            nn.Linear(192, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1)           # Outputs single channel (Pressure u)
        )

    def forward(self, invars, coords):
        """
        Args:
            invars: Permeability field tensor [B, 1, H, W]
            coords: Query physical coordinates tensor [B, 2, H, W]
        """
        B, _, H, W = invars.shape
        
        # Step 1: Extract continuous feature map via FNO
        # Output shape: [B, latent_channels, H, W]
        h_param_field = self.param_encoder(invars)
        
        # Step 2: Sample and align features spatially
        # Instead of global pooling, we preserve localized parameters. 
        # For a structured grid, we can directly permute or use grid_sample for meshless queries.
        h_param_flat = h_param_field.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        
        # Step 3: Process coordinate maps
        # Reshape coords to [B*H*W, 2] for the fully connected architecture
        coords_flat = coords.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        h_coord_flat = self.coord_encoder(coords_flat)
        
        # Step 4: Concatenate and decode point-wise
        h_concat = torch.cat([h_coord_flat, h_param_flat], dim=-1)
        pred_flat = self.manifold_net(h_concat)
        
        # Step 5: Reconstruction back to standard PhysicsNeMo grid shapes [B, C, H, W]
        pred_grid = pred_flat.view(B, H, W, -1).permute(0, 3, 1, 2)
        
        return pred_grid


# ============================================================
# Normalization Utilities
# ============================================================

def denormalize(x, mean, std):
    return x * std + mean


# ============================================================
# Main Physics-Informed Training Implementation
# ============================================================

def train_p2inn(cfg: DictConfig):
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Distributed Environment Setup
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    log = PythonLogger(name="p2inn_darcy")
    LaunchLogger.initialize()

    # Normalization metrics
    k_mean, k_std = cfg.normaliser.permeability.mean, cfg.normaliser.permeability.std_dev
    u_mean, u_std = cfg.normaliser.darcy.mean, cfg.normaliser.darcy.std_dev

    # Model Instantiation
    model = P2INN_Darcy(cfg).to(device)
    
    # Physics Informer
    phy_informer = get_physics_informer(device=device, equation=cfg.equation)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.start_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.gamma)
    
    # Setup Dataloader and Validation Engines via NeMo structures
    dataloader, validator = get_darcy_setup(cfg)
    
    steps_per_epoch = ceil(cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size)

    # ========================================================
    # STATIC CAPTURE TRAINING GRAPH
    # ========================================================
    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        logger=log,
        use_amp=False,
        use_graphs=False, # Set to True if shapes are perfectly static across iterations
    )
    def forward_train(invars_norm, coords):
        # 1. Forward Pass maintaining structured matrix footprints
        pred_norm = model(invars_norm, coords)
        
        # 2. Re-scale to physical domain
        k_phys = denormalize(invars_norm[:, 0:1], k_mean, k_std)
        u_phys = denormalize(pred_norm, u_mean, u_std)

        # 3. Physics Residual calculation using standard grid shapes
        residuals = phy_informer.forward(
            {
                "u": u_phys,
                "k": k_phys,
                "coordinates": coords,
            }
        )

        pde_residual = residuals["diffusion_u"]
        # Standard boundary trim for stable stencil computation
        pde_residual = pde_residual[:, :, 2:-2, 2:-2] 
        
        loss_pde = torch.mean(torch.square(pde_residual))
        return loss_pde

    # ========================================================
    # TRAINING LOOP
    # ========================================================
    log.success("Starting Physics-Compliant P2INN training...")
    
    for epoch in range(1, cfg.training.max_pseudo_epochs + 1):
        running_loss = 0.0
        
        with LaunchLogger("train", epoch=epoch, num_mini_batch=steps_per_epoch) as logger:
            for _, batch in zip(range(steps_per_epoch), dataloader):
                
                # Extract inputs
                invars = batch["permeability"].to(device) # [B, 1, H, W]
                B, _, H, W = invars.shape
                
                # Construct coordinate mesh in [B, 2, H, W] shape layout for NeMo tools
                x = torch.linspace(0, 1, H, device=device)
                y = torch.linspace(0, 1, W, device=device)
                grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
                coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

                # Execute static training pass
                loss = forward_train(invars, coords)
                
                loss_val = float(loss.detach().cpu())
                running_loss += loss_val
                logger.log_minibatch({"loss": loss_val})
                
            logger.log_epoch({"avg_train_loss": running_loss / steps_per_epoch})
            
        scheduler.step()
        
    return model