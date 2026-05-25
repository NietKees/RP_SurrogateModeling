import torch
import os
import hydra
import physicsnemo
import sys
from math import ceil
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physics_utils import get_physics_informer
from data_utils import get_darcy_setup
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from physicsnemo.utils import (
    StaticCaptureTraining,
    StaticCaptureEvaluateNoGrad,
    load_checkpoint,
    save_checkpoint,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def denormalize(x, mean, std):
    return x * std + mean


# @hydra.main(version_base="1.3", config_path=".", config_name="Darcy_PINN_config")
def train_pinn(cfg: DictConfig) -> None:
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    log = PythonLogger(name="corrected_pino")
    log.file_logging()
    LaunchLogger.initialize()


    print(f"Running on device: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Initialize Physics Informer via the factory function using autodiff
    physicsInformer = get_physics_informer(device, 'darcy', method="autodiff")

    # 2. Extract Normalization Constants
    k_mean, k_std = 1.25, 0.75
    u_mean, u_std = 4.52e-2, 2.79e-2

    # 3. Model Architecture
    model = FullyConnected(
        in_features=3,       # Inputs: x, y coordinate, and local k profile value
        out_features=1,      # Output: Normalized u
        num_layers=6,
        layer_size=cfg.arch.layer_size,
        activation_fn="silu" 
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    
    dataloader, validator = get_darcy_setup(cfg)

    ckpt_args = {
        "path": "./PINN/checkpoints",
        "models": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    # 4. Parametric Forward Training Step
    @StaticCaptureTraining(model=model, optim=optimizer, use_graphs=False)
    def forward_train(sampled_coords, sampled_k_norm, bc_coords, bc_k_norm):
        # Apply tracking explicitly on the sliced inputs inside the captured context
        sampled_coords = sampled_coords.clone().detach().requires_grad_(True)
        sampled_k_norm = sampled_k_norm.clone().detach().requires_grad_(True)

        # PDE Forward Pass
        input_tensor = torch.cat([sampled_coords, sampled_k_norm], dim=1)
        u_norm = model(input_tensor)

        # Critical: Convert to true physical scales before taking derivative operations
        u_phys = u_norm * u_std + u_mean
        k_phys = sampled_k_norm * k_std + k_mean

        residuals = physicsInformer.forward(
            {
                "coordinates": sampled_coords,
                "u": u_phys,
                "k": k_phys  # Note: your factory uses key 'k' or matching the Diffusion param
            }
        )
        
        # Pull output structural key dynamically matching your factory's output structure
        pde_key = list(residuals.keys())[0]
        loss_pde = torch.mean(torch.square(residuals[pde_key]))

        # Boundary Condition Forward Pass (Dirichlet 0 on the physical u scale)
        bc_input = torch.cat([bc_coords, bc_k_norm], dim=1)
        u_bc_norm = model(bc_input)
        u_bc_phys = u_bc_norm * u_std + u_mean
        loss_bc = torch.mean(torch.square(u_bc_phys))

        # Balanced Hybrid Loss Execution
        total_loss = loss_pde + (20.0 * loss_bc)
        return total_loss
    
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


    print("Parametric PINN started training")

    normaliser = { 
        "permeability": (k_mean, k_std), 
        "darcy": (u_mean, u_std),
    }
    # dataloader = Darcy2D(
    #     resolution=64, batch_size=32, nr_permeability_freq=5, normaliser=normaliser
    # )
    dataloader_iter = iter(dataloader)

    validation_iters = ceil(
        cfg.validation.sample_size /
        cfg.training.batch_size
    )

    current_val_error= float('inf')
    epoch = 1

    while epoch < cfg.training.max_epochs + 1 and current_val_error >= 0.005:
    # 5. Training Loop execution
    # for epoch in range(1, cfg.training.max_epochs + 1):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            
        perm = batch["permeability"].to(device) # Shape: [B, 1, H, W]
        B, _, H, W = perm.shape

        # Generate spatial layout templates
        x = torch.linspace(0, 1, H, device=device)
        y = torch.linspace(0, 1, W, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        # Spatial Coordinates and Permeability mapping arrays
        coords_all = torch.stack([grid_x, grid_y], dim=-1).repeat(B, 1, 1, 1) # [B, H, W, 2]
        k_all_norm = perm.permute(0, 2, 3, 1) # [B, H, W, 1]

        # --- Boundary Conditions Extractor ---
        # Mask and collect the 4 outer walls across the domain boundaries
        bc_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        bc_mask[0, :] = True
        bc_mask[-1, :] = True
        bc_mask[:, 0] = True
        bc_mask[:, -1] = True

        bc_coords = coords_all[:, bc_mask].view(-1, 2)
        bc_k_norm = k_all_norm[:, bc_mask].view(-1, 1)

        # --- Interior Domain Interior Collocation Sampler ---
        coords_flat = coords_all.view(-1, 2)
        k_flat_norm = k_all_norm.reshape(-1, 1)

        num_samples = cfg.training.num_boundary_points
        idx = torch.randperm(coords_flat.size(0), device=device)[:num_samples]
        
        # Execute the optimization step
        loss = forward_train(coords_flat[idx], k_flat_norm[idx], bc_coords, bc_k_norm)
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Total Multi-Parametric Loss: {loss.item():.6e} | LR: {optimizer.param_groups[0]['lr']:.3e}")

        if epoch % 100 == 0:
            # ckpt_path = os.path.join("checkpoints", f"checkpoint_{epoch}.pt")
            # torch.save(model.state_dict(), ckpt_path)
            # print(f"Saved checkpoint configuration: {ckpt_path}")
            save_checkpoint(**ckpt_args, epoch=epoch)
            log.success(f"Checkpoint saved at epoch {epoch}")

        # ===========================
        # validation
        # ===========================

        # if epoch % cfg.validation.validation_pseudo_epochs == 0:

            # with LaunchLogger("valid", epoch=epoch) as logger:
            #     total_l2 = 0.0

            #     for i, batch in zip(range(validation_iters), dataloader):
            #         invars = batch["permeability"].to(device)
            #         target = batch["darcy"].to(device)

            #         pred = forward_eval(invars)

            #         # Denormalize fields safely for accurate loss measurement
            #         k_phys = denormalize(invars[:, 0:1], k_mean, k_std)
            #         pred_phys = denormalize(pred, u_mean, u_std)
            #         target_phys = denormalize(target, u_mean, u_std)

            #         # BUG FIX: Use identical physical spaces inside validator 
            #         val_loss = validator.compare(
            #             k_phys,
            #             target_phys,
            #             pred_phys,
            #             i,  # Let GridValidator manage plotting internally via sample rank
            #             logger,
            #         )
                    
            #         # Safe cast handling for return objects
            #         total_l2 += float(val_loss)

            #     current_val_error = total_l2 / validation_iters
            #     logger.log_epoch({"relative_l2_physical": current_val_error})
                
            #     print(f"--- Epoch {epoch} | Combined GridValidator L2 Error: {current_val_error:.6f} ---")

            #     # Early stop check right after validation runs
            #     if current_val_error < 0.005:
            #         log.success(f"Target metric achieved! Error ({current_val_error:.5f}) < {0.005}")
            #         break
        # ========================================================
        # VALIDATION PIPELINE
        # ========================================================
        if epoch % cfg.validation.validation_pseudo_epochs == 0:
            # Re-enable the logger context so validation metrics print to terminal
            with LaunchLogger("PINN_valid", epoch=epoch) as logger:
                total_error = 0.0
                
                for i, batch in zip(range(validation_iters), dataloader):
                    invar = batch["permeability"].to(device)  # [B, 1, H, W]
                    target = batch["darcy"].to(device)        # [B, 1, H, W]
                    val_B, _, val_H, val_W = invar.shape
                    
                    x_test = torch.linspace(0, 1, val_H, device=device)
                    y_test = torch.linspace(0, 1, val_W, device=device)
                    grid_x_t, grid_y_t = torch.meshgrid(x_test, y_test, indexing='ij')
                    
                    # 1. Replicate exact training dimensions layout
                    coords_all_t = torch.stack([grid_x_t, grid_y_t], dim=-1).repeat(val_B, 1, 1, 1) # [B, H, W, 2]
                    k_all_norm_t = invar.permute(0, 2, 3, 1) # [B, H, W, 1]
                    
                    # 2. Flatten both consistently across the same dimensions
                    coords_flat_t = coords_all_t.view(-1, 2)
                    k_flat_norm_t = k_all_norm_t.reshape(-1, 1)
                    
                    with torch.no_grad():
                        input_eval = torch.cat([coords_flat_t, k_flat_norm_t], dim=1)
                        pred_flat = forward_eval(input_eval)
                        
                        # 3. Reshape cleanly back to matching [B, 1, H, W] grid structure
                        pred = pred_flat.view(val_B, val_H, val_W, 1).permute(0, 3, 1, 2)

                    # Pass directly to your validator module
                    loss = validator.compare(
                        invar, target, pred, 
                        step=i, logger=logger, 
                        title=f'validation_{epoch} PINN'
                    )
                    
                    relative_l2 = float(loss.detach().cpu().item())
                    total_error += relative_l2

                current_val_error = total_error / validation_iters
                logger.log_epoch({"relative_l2_physical": current_val_error})
                print(f"--- Epoch {epoch} | GridValidator L2 Error: {current_val_error:.6f} ---")
        epoch += 1
                  
    save_checkpoint(**ckpt_args, epoch=epoch - 1)
    log.success(f"PINO training complete. Final error achieved: {current_val_error:.5f}")

if __name__ == "__main__":
    train_pinn()

# import os
# import torch
# import hydra
# import pandas as pd
# import numpy as np
# from omegaconf import DictConfig

# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, ".."))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# from physicsnemo.distributed import DistributedManager
# from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
# from physicsnemo.models.mlp.fully_connected import FullyConnected
# from physicsnemo.utils.logging import LaunchLogger, PythonLogger
# from data_utils import get_darcy_setup
# from physics_utils import get_physics_informer
# from physicsnemo.utils import save_checkpoint

# # Optimize CUDA allocation strategy
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# def darcy_pinn_trainer(cfg: DictConfig) -> None:
#     # 1. Environment and Distributed Setup
#     os.environ["RANK"] = "0"
#     os.environ["WORLD_SIZE"] = "1"
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"

#     DistributedManager.initialize()
#     dist = DistributedManager()
#     device = dist.device

#     log = PythonLogger(name="pinn_trainer")
#     log.info(f"Running on device: {device}")
#     os.makedirs("checkpoints", exist_ok=True)

#     # 2. Normalization Constants (Accessed from Root Config)
#     k_mean = cfg.normaliser.permeability.mean
#     k_std = cfg.normaliser.permeability.std_dev
#     u_mean = cfg.normaliser.darcy.mean
#     u_std = cfg.normaliser.darcy.std_dev

#     # 3. Model Architecture (Parametric PINN MLP)
#     model = FullyConnected(
#         in_features=3,       # x, y, and local permeability k
#         out_features=1,      # u (pressure)
#         num_layers=6,
#         layer_size=512,
#         activation_fn="silu" 
#     ).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#     # 4. Data Pipeline Setup 
#     dataloader, validator = get_darcy_setup(cfg)
#     dataloader_iter = iter(dataloader)

#     physics_informer = get_physics_informer(device, cfg.equation)

#     log.info("PINN training started with VRAM Optimization + Random Collocation...")


#     ckpt_args = {
#         "path": "./PINN/checkpoints",
#         "models": model,
#         "optimizer": optimizer,
#         # "scheduler": scheduler,
#     }

#     epoch = 1 #max(1, loaded_epoch + 1)
#     current_val_error = float('inf')
#     target_error_threshold = cfg.training.target_error_threshold

#     # 5. Main Training Loop
#     #for epoch in range(1, cfg.training.max_epochs + 1):
#     while epoch < cfg.trainig.max_epochs and current_val_error >= target_error_threshold:
#         try:
#             batch = next(dataloader_iter)
#         except StopIteration:
#             dataloader_iter = iter(dataloader)
#             batch = next(dataloader_iter)
            
#         perm = batch["permeability"].to(device)  # [B, 1, H, W]
#         B, _, H, W = perm.shape

#         # Generate coordinate grid space 
#         x = torch.linspace(0, 1, H, device=device)
#         y = torch.linspace(0, 1, W, device=device)
#         grid_x, grid_y = torch.meshgrid(x, y, indexing='ij') # Shape: [H, W]

#         # 1. Create a structured 4D coordinates grid: [B, 2, H, W]
#         coords_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
#         coords_grid.requires_grad_(True)

#         # 2. Flatten grid layout temporarily *only* for the MLP forward pass
#         coords_flat = coords_grid.permute(0, 2, 3, 1).reshape(-1, 2) # [B*H*W, 2]
#         perm_flat = perm.permute(0, 2, 3, 1).reshape(-1, 1)        # [B*H*W, 1]

#         model.train()
#         optimizer.zero_grad()

#         # Forward pass through MLP (Normalized domain)
#         input_tensor = torch.cat([coords_flat, perm_flat], dim=1) # [B*H*W, 3]
#         u_norm_flat = model(input_tensor)                         # [B*H*W, 1]

#         # 3. Reshape MLP predictions back to a structured 4D grid: [B, 1, H, W]
#         u_norm = u_norm_flat.view(B, H, W, 1).permute(0, 3, 1, 2)
        
#         # De-normalize values back to physical scales
#         u_phys = u_norm * u_std + u_mean
#         k_phys = perm * k_std + k_mean

#         # --- USING THE PHYSICS INFORMER (Now receives matching 4D tensors) ---
#         physics_outputs = physics_informer.forward({
#             "coordinates": coords_grid,  # [B, 2, H, W]
#             "u": u_phys,                 # [B, 1, H, W]
#             "k": k_phys                  # [B, 1, H, W]
#         })
        
#         # Extract your targeted PDE residual term
#         pde_residual = physics_outputs["diffusion_u"]
#         pde_loss = torch.mean(torch.square(pde_residual))
        
#         # --- Elegant & Fast Boundary Conditions on 4D Grids ---
#         # Instead of sampling random points, slice the actual edges of your 4D grid walls
#         bc_loss = (
#             torch.mean(torch.square(u_phys[:, :, :, 0])) +   # Left boundary (x=0)
#             torch.mean(torch.square(u_phys[:, :, :, -1])) +  # Right boundary (x=1)
#             torch.mean(torch.square(u_phys[:, :, 0, :])) +   # Top boundary (y=0)
#             torch.mean(torch.square(u_phys[:, :, -1, :]))    # Bottom boundary (y=1)
#         ) / 4.0

#         # Total Loss Balance
#         loss = pde_loss + 20.0 * bc_loss

#         loss.backward()
#         optimizer.step()

#         if epoch % 10 == 0:
#             log.info(f"Epoch {epoch} | Total Loss: {loss.item():.6e} | PDE: {pde_loss.item():.4e} | BC: {bc_loss.item():.4e}")

#         if epoch % cfg.training.rec_results_freq == 0:
#             save_checkpoint(**ckpt_args, epoch=epoch)
#             log.success(f"Checkpoint saved at epoch {epoch}")
#             torch.cuda.empty_cache()

#         epoch += 1
#         current_val_error = loss
#         # if epoch % 100 == 0:
#         #     ckpt_path = os.path.join("checkpoints", f"checkpoint_{epoch}.pt")
#         #     torch.save(model.state_dict(), ckpt_path)
#         #     torch.cuda.empty_cache()

#     torch.save(model.state_dict(), "darcy_PINN.pt")
#     log.success("Training Complete!")