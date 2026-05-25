import os
import sys
from math import ceil
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


import hydra
import torch
import torch.nn.functional as F

from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from physicsnemo.utils import (
    StaticCaptureTraining,
    StaticCaptureEvaluateNoGrad,
    load_checkpoint,
    save_checkpoint,
)

from data_utils import get_darcy_setup
from physics_utils import get_physics_informer
from validator import GridValidator


# ============================================================
# Utilities
# ============================================================

def denormalize(x, mean, std):
    return x * std + mean


def normalize(x, mean, std):
    return (x - mean) / std


# ============================================================
# Main Training
# ============================================================

def train_pifno(cfg: DictConfig):

    # --------------------------------------------------------
    # Distributed setup
    # --------------------------------------------------------

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log = PythonLogger(name="corrected_pino")
    log.file_logging()
    LaunchLogger.initialize()

    # --------------------------------------------------------
    # Normalization stats
    # --------------------------------------------------------

    k_mean = cfg.normaliser.permeability.mean
    k_std = cfg.normaliser.permeability.std_dev

    u_mean = cfg.normaliser.darcy.mean
    u_std = cfg.normaliser.darcy.std_dev

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FNO(
        in_channels=cfg.arch.in_channels,
        out_channels=cfg.arch.out_features,
        decoder_layers=cfg.arch.decoder_layers,
        decoder_layer_size=cfg.arch.decoder_layer_size,
        dimension=cfg.arch.dimension,
        latent_channels=cfg.arch.latent_channels,
        num_fno_layers=cfg.arch.num_fno_layers,
        num_fno_modes=cfg.arch.num_fno_modes,
        padding=cfg.arch.padding,
    ).to(device)

    # --------------------------------------------------------
    # Physics informer
    # --------------------------------------------------------

    phy_informer = get_physics_informer(
        device=device,
        equation=cfg.equation,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.start_lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.training.gamma,
    )

    # This drops the LR by 15% (multiplies by 0.85) every 8 epochs
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, 
    #     step_size=cfg.scheduler.decay_pseudo_epochs, 
    #     gamma=cfg.scheduler.decay_rate
    # )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    dataloader, validator = get_darcy_setup(cfg)

    # --------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------

    ckpt_args = {
        "path": "./PINN/PiFNO/checkpoints",
        "models": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

    loaded_epoch = load_checkpoint(device=device, **ckpt_args)

    # --------------------------------------------------------
    # Pseudo epoch setup
    # --------------------------------------------------------

    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size /
        cfg.training.batch_size
    )

    validation_iters = ceil(
        cfg.validation.sample_size /
        cfg.training.batch_size
    )

    # ========================================================
    # TRAIN STEP
    # ========================================================

    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        logger=log,
        use_amp=False,
        use_graphs=False,
    )
    def forward_train(invars_norm, target_norm):

        pred_norm = model(invars_norm)
        # loss_data = F.mse_loss(pred_norm, target_norm)

        k_phys = denormalize(invars_norm[:, 0:1], k_mean, k_std)
        u_phys = denormalize(pred_norm, u_mean, u_std)

        B, _, H, W = u_phys.shape
        x = torch.linspace(0, 1, H, device=device)
        y = torch.linspace(0, 1, W, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        residuals = phy_informer.forward(
            {
                "u": u_phys,
                "k": k_phys,
                "coordinates": coords,
            }
        )

        pde_residual = residuals["diffusion_u"]
        pde_residual = pde_residual[:, :, 2:-2, 2:-2]
        loss_pde = torch.mean(torch.abs(pde_residual))
        print(f"physics:{loss_pde}") # , data loss: {loss_data}")
        physics_weight = cfg.physics.weight * 1/250
        loss = physics_weight * loss_pde # +loss_data
        return loss

    # ========================================================
    # EVAL STEP
    # ========================================================

    @StaticCaptureEvaluateNoGrad(
        model=model,
        logger=log,
        use_amp=False,
        use_graphs=False,
    )
    def forward_eval(invars):
        return model(invars)

    # ========================================================
    # Training loop (Modified for conditional termination)
    # ========================================================

    if loaded_epoch == 0:
        log.success("Starting PINO training...")
    else:
        log.warning(f"Resuming from epoch {loaded_epoch}")

    pseudo_epoch = max(1, loaded_epoch + 1)
    current_val_error = float('inf')
    target_error_threshold = cfg.training.target_error_threshold

    # Run until error is met, or an absolute safety cap is reached to avoid deadlocks
    while current_val_error >= target_error_threshold and pseudo_epoch <= cfg.training.max_pseudo_epochs:

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        with LaunchLogger(
            "train",
            epoch=pseudo_epoch,
            num_mini_batch=steps_per_pseudo_epoch,
            epoch_alert_freq=1,
        ) as logger:

            running_loss = 0.0

            for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
                invars = batch["permeability"].to(device)
                target = batch["darcy"].to(device)

                loss = forward_train(invars, target)
                loss_value = float(loss.detach().cpu())
                running_loss += loss_value

                logger.log_minibatch({"loss": loss_value})

            avg_loss = running_loss / steps_per_pseudo_epoch
            logger.log_epoch(
                {
                    "avg_train_loss": avg_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:

            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_l2 = 0.0

                for i, batch in zip(range(validation_iters), dataloader):
                    invars = batch["permeability"].to(device)
                    target = batch["darcy"].to(device)

                    pred = forward_eval(invars)

                    # Denormalize fields safely for accurate loss measurement
                    k_phys = denormalize(invars[:, 0:1], k_mean, k_std)
                    pred_phys = denormalize(pred, u_mean, u_std)
                    target_phys = denormalize(target, u_mean, u_std)

                    # BUG FIX: Use identical physical spaces inside validator 
                    val_loss = validator.compare(
                        k_phys,
                        target_phys,
                        pred_phys,
                        i,  # Let GridValidator manage plotting internally via sample rank
                        logger,
                    )
                    
                    # Safe cast handling for return objects
                    total_l2 += float(val_loss)

                current_val_error = total_l2 / validation_iters
                logger.log_epoch({"relative_l2_physical": current_val_error})
                
                print(f"--- Epoch {pseudo_epoch} | Combined GridValidator L2 Error: {current_val_error:.6f} ---")

                # Early stop check right after validation runs
                if current_val_error < target_error_threshold:
                    log.success(f"Target metric achieved! Error ({current_val_error:.5f}) < {target_error_threshold}")
                    break

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        # if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
        scheduler.step()

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)
            log.success(f"Checkpoint saved at epoch {pseudo_epoch}")

        pseudo_epoch += 1

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    save_checkpoint(**ckpt_args, epoch=pseudo_epoch - 1)
    log.success(f"PINO training complete. Final error achieved: {current_val_error:.5f}")