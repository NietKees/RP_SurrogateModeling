# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from math import ceil

import torch
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np

from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, LaunchLogger
from Burgers.generator import burgers_generator, burgers_physics_residual

# UPDATED: Pointing to your dataset factories for the Burgers sequence
from data_utils import get_burgers_setup 
@hydra.main(version_base="1.3", config_path="..", config_name="Burgers_pipeline_config.yaml")
def burgers_fno_trainer(cfg: DictConfig) -> None:
    """Training for the 1D+Time Burgers equation benchmark problem inspired by neuraloperator.
    
    The FNO handles the problem by evaluating inputs shaped as (B, in_channels, S, T) 
    and outputting the full field predictions matching the target space-time configuration.
    """    
    

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

    # Initialize monitoring logs
    log = PythonLogger(name="burgers_pino")
    log.file_logging()
    LaunchLogger.initialize()  

    # Define model, loss, optimizer, and scheduler
    # NOTE: Even though Burgers is 1D physically, dimension=2 because space and time 
    # form a 2D mesh layout (Size_x, Size_t).
    model = FNO(
        in_channels=cfg.arch.in_channels,
        out_channels=cfg.arch.out_features,
        decoder_layers=cfg.arch.decoder_layers,
        decoder_layer_size=cfg.arch.decoder_layer_size,
        dimension=cfg.arch.dimension,
        latent_channels=cfg.arch.latent_channels,
        num_fno_layers=cfg.arch.num_fno_layers,
        num_fno_modes=[16, 12],  # cfg.arch.num_fno_modes,
        padding=cfg.arch.padding,
    ).to(device)
    
    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.gamma)

    # Load dataset dataloaders
    dataloader, validation_dataloader, validator = get_burgers_setup(cfg)
    
    ckpt_args = {
        "path": f"./PINO/Burger_checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = load_checkpoint(device=device, **ckpt_args)

    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    nx = getattr(cfg.training, "nx", 128)
    nt = getattr(cfg.training, "nt", 100)
    nu = getattr(cfg.physics, "nu", 0.01)
    tmax = getattr(cfg.physics, "tmax", 1.0)
    
    dx = (2.0 * np.pi) / nx
    dt = tmax / (nt - 1)
    # ========================================================
    # TRAINING STEP
    # ========================================================
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target):
        if len(invars.shape) == 3:
            T_steps = target.shape[-1]
            invars = invars.unsqueeze(-1).repeat(1, 1, 1, T_steps)
            
        pred = model(invars)
        
        # 1. Supervised Data Loss
        loss_data = loss_fun(pred, target)
        
        # 2. Physics-Informed Unsupervised Residual Loss
        # pde_residual = burgers_physics_residual(pred, nu, dx, dt)
        pde_residual = burgers_physics_residual(pred, nu=cfg.training.nu, dx_val=dx, dt_val=dt)
        loss_pde = F.mse_loss(pde_residual, torch.zeros_like(pde_residual))

        sol_pde_residual = burgers_physics_residual(target, nu=cfg.training.nu, dx_val=dx, dt_val=dt)
        sol_loss = F.mse_loss(sol_pde_residual, torch.zeros_like(sol_pde_residual))

        # Combine Loss using weight scales from configurations
        physics_weight = getattr(cfg.physics, "weight", 0.1) # * 1/nx
        print(f'data loss: {loss_data},residual loss:{loss_pde}, real_residual_loss: {sol_loss}')
        # Total combined optimization target
        loss = loss_data +  physics_weight * loss_pde
        return loss

    # ========================================================
    # EVALUATION STEP
    # ========================================================
    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars, target_shape_t=None):
        if len(invars.shape) == 3 and target_shape_t is not None:
            invars = invars.unsqueeze(-1).repeat(1, 1, 1, target_shape_t)
        return model(invars)

    if loaded_pseudo_epoch == 0:
        log.success("Burgers PINO Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch + 1}.")
    
    pseudo_epoch = max(1, loaded_pseudo_epoch + 1)
    current_val_error = float('inf')
    target_error_threshold = cfg.training.target_error_threshold
    
    # Core Loop Execution
    while current_val_error >= target_error_threshold and pseudo_epoch <= cfg.training.max_pseudo_epochs + 1:
        
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
                # Target names transformed to reflect traditional Burgers datasets
                # (v_init: initial condition function, v: full solution tensor matrix)
                x_in = batch["v_init"].to(device)
                y_target = batch["v"].to(device)
                
                loss = forward_train(x_in, y_target)
                logger.log_minibatch({"loss": loss.detach()})
                
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # Checkpoint Intervals
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # Validation Step
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                for _, batch in zip(range(validation_iters), validation_dataloader):
                    x_in = batch["v_init"].to(device)
                    y_target = batch["v"].to(device)
                    
                    pred_out = forward_eval(x_in, target_shape_t=y_target.shape[-1])
                    T_steps = y_target.shape[-1]
                    x_in_4d = x_in.unsqueeze(-1).repeat(1, 1, 1, T_steps)
                    val_loss = validator.compare(
                        x_in_4d,
                        y_target,
                        pred_out,
                        pseudo_epoch,
                        logger,
                        title=f'Burgers_PINO_val_epoch_{pseudo_epoch}'
                    )
                    total_loss += val_loss
                    
                current_val_error = total_loss / validation_iters
                logger.log_epoch({"Validation error": current_val_error})
                
        scheduler.step()
        pseudo_epoch += 1

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs + 1)
    log.success("Burgers PINO Training completed successfully!")


if __name__ == "__main__":
    burgers_fno_trainer()