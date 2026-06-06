# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import hydra
from omegaconf import DictConfig, OmegaConf
from math import ceil

import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, LaunchLogger
from physicsnemo.core.module import Module
from dataclasses import dataclass
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.models.mlp.fully_connected import FullyConnected

from data_utils import get_burgers_setup 


@dataclass
class P2INNMetaData(ModelMetaData):
    name: str = "P2INN_Burgers"
    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    amp_gpu: bool = True  

# -------------------------------------------------------------------------
# 2. FIXED PARAMETRIC PINN ARCHITECTURE
# -------------------------------------------------------------------------
class ParametricPINN(Module):
    meta = P2INNMetaData()
    
    # FIX: Avoid conflicting with framework inspection keys by specifying explicit layout dimensions
    def __init__(self, coord_dim=2, param_dim=1, hidden_dim_c=64, hidden_dim_p=64, out_features=1):
        """
        Parametric PINN with separate latent representations for 
        coordinates (x, t) and global physics parameters (nu).
        """
        # Forward metadata initialization explicitly
        super().__init__(meta=self.meta)
        
        # Coordinate Trunk: Processes spatial and temporal points (x, t)
        self.g_c = FullyConnected(
            in_features=coord_dim, 
            out_features=hidden_dim_c, 
            num_layers=3, 
            layer_size=hidden_dim_c
        )
        
        # Parameter Trunk: Tailored to capture varying fluid viscosity regimes (nu)
        self.g_p = FullyConnected(
            in_features=param_dim, 
            out_features=hidden_dim_p, 
            num_layers=3, 
            layer_size=hidden_dim_p
        )
        
        # Manifold / Merging Layers: Decodes the combined spaces
        self.g_g = FullyConnected(
            in_features=hidden_dim_c + hidden_dim_p, 
            out_features=out_features, 
            num_layers=4, 
            layer_size=hidden_dim_c
        )
        
    def forward(self, x, t, nu):
        # Stitch independent space-time vectors together [N, 2]
        coords = torch.cat([x, t], dim=-1)
        
        # Process structural representations
        h_coord = self.g_c(coords)
        h_param = self.g_p(nu)
        
        # Join tensors along the hidden dimension interface
        h_concat = torch.cat([h_coord, h_param], dim=-1)
        return self.g_g(h_concat)

@hydra.main(version_base="1.3", config_path="..", config_name="Burgers_pipeline_config.yaml")
def burgers_p2inn_trainer(cfg: DictConfig) -> None:
    container = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(container)
    if cfg and hasattr(cfg, 'pino'):
        cfg = cfg.pino

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    DistributedManager.initialize()  
    dist = DistributedManager()  
    device = dist.device

    log = PythonLogger(name="burgers_p2inn")
    log.file_logging()
    LaunchLogger.initialize()  

    model = ParametricPINN(
        coord_dim=2,       # (x, t)
        param_dim=1,       # (nu)
        hidden_dim_c=cfg.arch.decoder_layer_size, # map your yaml keys here
        hidden_dim_p=cfg.arch.decoder_layer_size,
        out_features=cfg.arch.out_features
    ).to(device)
    
    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.gamma)

    dataloader, validation_dataloader, validator = get_burgers_setup(cfg)
    
    ckpt_args = {
        "path": f"./P2INN/Burger_checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = load_checkpoint(device=device, **ckpt_args)

    steps_per_pseudo_epoch = ceil(cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size)
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    
    nx = getattr(cfg.training, "nx", 128)
    nt = getattr(cfg.training, "nt", 100)
    nu_default = getattr(cfg.training, "nu", 0.01)
    tmax = getattr(cfg.physics, "tmax", 1.0)
    
    # Initialize basic grid mesh coordinates
    x_coords = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    t_coords = np.linspace(0.0, tmax, nt)
    X_mesh, T_mesh = np.meshgrid(x_coords, t_coords, indexing='ij')

    # Convert coordinates to flat tracking shapes
    x_flat = torch.tensor(X_mesh.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
    t_flat = torch.tensor(T_mesh.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)

    # ========================================================
    # PHYSICS-NEMO COMPATIBLE TRAINING STEP
    # ========================================================
    # ========================================================
    # MEMORY-OPTIMIZED P2INN TRAINING STEP
    # ========================================================
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(target_solutions, nu_tensor):
        batch_size = target_solutions.shape[0]
        total_grid_points = nx * nt
        
        # --- 1. DATA LOSS (Processed point-wise or downsampled) ---
        x_batch_data = x_flat.repeat(batch_size, 1).detach().clone()
        t_batch_data = t_flat.repeat(batch_size, 1).detach().clone()
        
        if nu_tensor.dim() == 1 or nu_tensor.shape[0] != batch_size:
            nu_batch_data = torch.full_like(x_batch_data, nu_default)
        else:
            nu_batch_data = nu_tensor.view(-1, 1, 1).repeat(1, total_grid_points, 1).view(-1, 1).detach().clone()
            
        y_target_flat = target_solutions.view(-1, 1)
        pred_flat_data = model(x_batch_data, t_batch_data, nu_batch_data)
        loss_data = loss_fun(pred_flat_data, y_target_flat)
        
        # --- 2. PHYSICS LOSS (Subsampled to prevent CUDA OOM) ---
        # Select a manageable number of random points across the domain (e.g., 5000-10000 points)
        num_collocation_points = 4096  # Adjust downward if you still hit OOM
        
        # Sample random indices from the tiled batch-space
        total_points = batch_size * total_grid_points
        idx = torch.randint(0, total_points, (num_collocation_points,), device=device)
        
        # Extract and isolate coordinates for the physics graph tracking step
        x_physics = x_batch_data.view(-1, 1)[idx].detach().clone()
        t_physics = t_batch_data.view(-1, 1)[idx].detach().clone()
        nu_physics = nu_batch_data.view(-1, 1)[idx].detach().clone()
        
        # Explicitly enable tracking on sub-sampled tensors
        x_physics.requires_grad_(True)
        t_physics.requires_grad_(True)
        
        # Forward pass on physics points only
        pred_physics = model(x_physics, t_physics, nu_physics)
        
        # Continuous Autograd Loop
        u_x = torch.autograd.grad(
            pred_physics, x_physics, 
            grad_outputs=torch.ones_like(pred_physics), 
            create_graph=True, retain_graph=True
        )[0]
        
        u_t = torch.autograd.grad(
            pred_physics, t_physics, 
            grad_outputs=torch.ones_like(pred_physics), 
            create_graph=True, retain_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x_physics, 
            grad_outputs=torch.ones_like(u_x), 
            create_graph=True, retain_graph=True
        )[0]
        
        # PDE Residual Formulation
        pde_residual = u_t + pred_physics * u_x - nu_physics * u_xx
        loss_pde = torch.mean(pde_residual ** 2)

        physics_weight = getattr(cfg.physics, "weight", 0.1)

        # Inside your training loop:
        # Start with purely supervised data training, then scale physics weight up
        if pseudo_epoch < 10:
            physics_weight = 0.0
        elif pseudo_epoch < 20:
            physics_weight = getattr(cfg.physics, "weight", 0.1) * 0.1
        else:
            physics_weight = getattr(cfg.physics, "weight", 0.1)
            
        loss = loss_data + physics_weight * loss_pde
        return loss
    # ========================================================
    # PHYSICS-NEMO COMPATIBLE EVALUATION STEP
    # ========================================================
    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(batch_size, nu_tensor):
        x_batch = x_flat.repeat(batch_size, 1)
        t_batch = t_flat.repeat(batch_size, 1)
        
        if nu_tensor.dim() == 1 or nu_tensor.shape[0] != batch_size:
            nu_batch = torch.full_like(x_batch, nu_default)
        else:
            nu_batch = nu_tensor.view(-1, 1, 1).repeat(1, nx * nt, 1).view(-1, 1)
            
        pred_flat = model(x_batch, t_batch, nu_batch)
        return pred_flat.view(batch_size, 1, nx, nt)

    # -------------------------------------------------------------------------
    # MAIN EXECUTION LOOP
    # -------------------------------------------------------------------------
    if loaded_pseudo_epoch == 0:
        log.success("Burgers P2INN Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch + 1}.")
    
    pseudo_epoch = max(1, loaded_pseudo_epoch + 1)
    current_val_error = float('inf')
    target_error_threshold = cfg.training.target_error_threshold
    
    while current_val_error >= target_error_threshold and pseudo_epoch <= cfg.training.max_pseudo_epochs + 1:
        
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
                y_target = batch["v"].to(device)      
                nu_tensor = batch["nu"].to(device)    
                
                loss = forward_train(y_target, nu_tensor)
                logger.log_minibatch({"loss": loss.detach()})
                
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # Validation Step
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                for _, batch in zip(range(validation_iters), validation_dataloader):
                    y_target = batch["v"].to(device)
                    nu_tensor = batch["nu"].to(device)
                    b_size = y_target.shape[0]
                    
                    pred_out = forward_eval(b_size, nu_tensor)
                    
                    x_in_fake = batch["v_init"].to(device).unsqueeze(-1).repeat(1, 1, 1, nt)
                    val_loss = validator.compare(
                        x_in_fake, y_target, pred_out,
                        pseudo_epoch, logger,
                        title=f'Burgers_P2INN_val_epoch_{pseudo_epoch}'
                    )
                    total_loss += val_loss
                    
                current_val_error = total_loss / validation_iters
                logger.log_epoch({"Validation error": current_val_error})
                
        scheduler.step()
        pseudo_epoch += 1

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs + 1)
    log.success("Burgers P2INN Training completed successfully!")

if __name__ == "__main__":
    burgers_p2inn_trainer()