import os
import torch
import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from physicsnemo.utils import StaticCaptureEvaluateNoGrad, load_checkpoint
from data_utils import get_darcy_setup
# Optimize CUDA allocation strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def load_model_weights(model, checkpoint_dir, device, log):
    """Utility to load checkpoint weights into a model instance."""
    # load_checkpoint expects 'checkpoint.pt' inside the dir. 
    # Ensure your PINO and FNO dirs each have a 'checkpoint.pt'
    try:
        load_checkpoint(path=checkpoint_dir, models=model, device=device)
        model.eval()
        log.success(f"Successfully loaded weights from {checkpoint_dir}")
    except Exception as e:
        log.error(f"Failed to load weights from {checkpoint_dir}: {e}")


@hydra.main(version_base="1.3", config_path=".", config_name="pipeline_config")
def darcy_pinn_trainer(cfg: DictConfig) -> None:
    # 1. Environment and Distributed Setup
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    log = PythonLogger(name="pinn_trainer")
    log.info(f"Running on device: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # 2. Normalization Constants
    k_mean, k_std = 1.25, 0.75
    u_mean, u_std = 4.52e-2, 2.79e-2
    normaliser = {
        "permeability": (k_mean, k_std), 
        "darcy": (u_mean, u_std),
    }

    # 3. Model Architecture (Parametric PINN MLP)
    model = FullyConnected(
        in_features=3,       # x, y, and local permeability k
        out_features=1,      # u (pressure)
        num_layers=6,
        layer_size=512,
        activation_fn="silu" 
    ).to(device)
    
    load_model_weights(model, "./PINN/checkpoints", device, log)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 4. Data Pipeline Setup (Training - Lowered Batch Size to save VRAM)
    train_resolution = 64
    dataloader = Darcy2D(
        resolution=train_resolution, batch_size=2, nr_permeability_freq=5, normaliser=normaliser, device=device
    )
    
    dataloader_iter = iter(dataloader)

    log.info("PINN training started with VRAM Optimization + Random Collocation...")

    # 5. Main Training Loop
    # for epoch in range(1, cfg.pinn.training.max_epochs + 1):
    #     try:
    #         batch = next(dataloader_iter)
    #     except StopIteration:
    #         dataloader_iter = iter(dataloader)
    #         batch = next(dataloader_iter)
            
    #     perm = batch["permeability"].to(device)  # [B, 1, H, W]
    #     B, _, H, W = perm.shape

    #     # Generate coordinate grid space 
    #     x = torch.linspace(0, 1, H, device=device)
    #     y = torch.linspace(0, 1, W, device=device)
    #     grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    #     coords_full = torch.stack([grid_x, grid_y], dim=-1).repeat(B, 1, 1, 1).view(-1, 2)
    #     a_vals_full = perm.permute(0, 2, 3, 1).reshape(-1, 1)

    #     # VRAM FIX: Sub-sample collocation points rather than processing the entire grid
    #     # 1024 points per batch item drastically lowers backpropagation memory footprints
    #     num_collocation = 1024 * B 
    #     idx_interior = torch.randperm(coords_full.size(0), device=device)[:num_collocation]
        
    #     coords = coords_full[idx_interior].detach().clone()
    #     coords.requires_grad_(True)  # Enable tracking on spatial coordinates subset
    #     a_vals = a_vals_full[idx_interior]

    #     model.train()
    #     optimizer.zero_grad()

    #     # Forward pass through parametric network
    #     input_tensor = torch.cat([coords, a_vals], dim=1)
    #     u_norm = model(input_tensor)

    #     # De-normalize back to physical dimensions for physics calculation
    #     u_phys = u_norm * u_std + u_mean
    #     k_phys = a_vals * k_std + k_mean

    #     # --- EXACT AUTOGRAD FOR Darcy Equation ---
    #     du_dxy = torch.autograd.grad(
    #         outputs=u_phys,
    #         inputs=coords,
    #         grad_outputs=torch.ones_like(u_phys),
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True
    #     )[0]
        
    #     du_dx = du_dxy[:, 0:1]
    #     du_dy = du_dxy[:, 1:2]

    #     flux_x = k_phys * du_dx
    #     flux_y = k_phys * du_dy

    #     dflux_dx = torch.autograd.grad(
    #         outputs=flux_x,
    #         inputs=coords,
    #         grad_outputs=torch.ones_like(flux_x),
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True
    #     )[0][:, 0:1]

    #     dflux_dy = torch.autograd.grad(
    #         outputs=flux_y,
    #         inputs=coords,
    #         grad_outputs=torch.ones_like(flux_y),
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True
    #     )[0][:, 1:2]

    #     pde_residual = dflux_dx + dflux_dy + 1.0
    #     pde_loss = torch.mean(torch.square(pde_residual))

    #     # --- Memory-Safe Boundary Conditions Enforcement ---
    #     # Explicitly sample points located along the walls of the domain
    #     boundary_mask_full = (coords_full[:, 0] == 0) | (coords_full[:, 0] == 1) | \
    #                         (coords_full[:, 1] == 0) | (coords_full[:, 1] == 1)
    #     coords_bc_raw = coords_full[boundary_mask_full]
    #     a_vals_bc_raw = a_vals_full[boundary_mask_full]
        
    #     idx_bc = torch.randperm(coords_bc_raw.size(0), device=device)[:256]
    #     coords_bc = coords_bc_raw[idx_bc]
    #     a_vals_bc = a_vals_bc_raw[idx_bc]

    #     input_tensor_bc = torch.cat([coords_bc, a_vals_bc], dim=1)
    #     u_norm_bc = model(input_tensor_bc)
    #     u_phys_bc = u_norm_bc * u_std + u_mean
    #     bc_loss = torch.mean(torch.square(u_phys_bc))

    #     # Total integrated balanced loss
    #     loss = pde_loss + 20.0 * bc_loss

    #     loss.backward()
    #     optimizer.step()
        
    #     if epoch % 10 == 0:
    #         log.info(f"Epoch {epoch} | Total Loss: {loss.item():.6e} | PDE: {pde_loss.item():.4e} | BC: {bc_loss.item():.4e}")

    #     if epoch % 100 == 0:
    #         ckpt_path = os.path.join("checkpoints", f"checkpoint_{epoch}.pt")
    #         torch.save(model.state_dict(), ckpt_path)
    #         # Flush out unreferenced cache memory maps to prevent fragmentation OOMs
    #         torch.cuda.empty_cache()

    torch.save(model.state_dict(), "darcy_PINN.pt")
    log.success("Training Complete! Starting Out-of-Distribution Evaluation Pipeline...")

    # ==================================================================
    # 6. Out-of-Distribution Scenario Testing Framework (Remains Mesh-Complete)
    # ==================================================================
    scenarios = [
        {"name": "Standard (Train)", "k": [0.5, 2.0], "freq": 5, "res": 64},
        {"name": "light Clay", "k": [1e-2, 1e-1], "freq": 5, "res": 64},
        # {"name": "Medium Clay", "k": [1e-3, 1e-2], "freq": 5, "res": 64},
        # {"name": "Extreme Clay", "k": [1e-7, 1e-6], "freq": 5, "res": 64},
        # {"name": "Medium Flow gravel", "k": [2.0, 5.0], "freq": 5, "res": 64},
        # {"name": "High Flow Gravel", "k": [5.0, 10.0], "freq": 5, "res": 64},
        # {"name": "High Complexity", "k": [0.5, 2.0], "freq": 25, "res": 64},
        # {"name": "ZeroShot_SuperRes_128", "k": [0.5, 2.0], "freq": 5, "res": 128},
        # {"name": "ZeroShot_SuperRes_256", "k": [0.5, 2.0], "freq": 5, "res": 256},
    ]

    all_results = []
    num_test_samples = cfg.test_samples 
    model.eval()

    @StaticCaptureEvaluateNoGrad(model=model, logger=log, use_amp=False, use_graphs=False)
    def forward_eval(invars):
        return model(invars)
    
    for sc in scenarios:
        log.info(f"Evaluating Scenario: {sc['name']} on PINN Architecture")
        
        test_dataloader = Darcy2D(
            resolution=sc['res'],
            batch_size=1,  
            min_permeability=sc['k'][0],
            max_permeability=sc['k'][1],
            nr_permeability_freq=sc['freq'],
            normaliser=normaliser,
            device=device
        )
        _, validator = get_darcy_setup(cfg)
        
        scenario_errors = []
        
        with LaunchLogger(f"PINN_{sc['name']}") as logger:
            for i, batch in zip(range(num_test_samples), test_dataloader):
                invar = batch["permeability"].to(device)  
                target = batch["darcy"].to(device)        
                
                _, _, H, W = invar.shape
                
                x_test = torch.linspace(0, 1, H, device=device)
                y_test = torch.linspace(0, 1, W, device=device)
                grid_x_t, grid_y_t = torch.meshgrid(x_test, y_test, indexing='ij')
                
                coords_t = torch.stack([grid_x_t, grid_y_t], dim=-1).view(-1, 2)
                a_vals_t = invar.permute(0, 2, 3, 1).reshape(-1, 1)
                
                with torch.no_grad():
                    input_eval = torch.cat([coords_t, a_vals_t], dim=1)
                    # pred_flat = model(input_eval)
                    pred_flat = forward_eval(input_eval)
                    pred = pred_flat.view(1, H, W).unsqueeze(1)

                loss = validator.compare(invar, target, pred, step=i, logger=logger, title=f'{sc}, PINN')
                # num_diff = torch.norm(pred - target)
                # den_diff = torch.norm(target)
                # relative_l2 = float((num_diff / den_diff).cpu().item())
                relative_l2 = float(loss.detach().cpu().item())
                
                scenario_errors.append(relative_l2)
                
                # if i == 0:
                #     try:
                #         loss_f = torch.nn.MSELoss(reduction="mean")
                #         # temp_validator = GridValidator(loss_fun=loss_f, norm=normaliser)
                #         validator.compare(invar, target, pred, step=1, logger=logger, title=f'{sc}, PINN')
                #     except Exception as e:
                #         log.warning(f"Skipping visualization: {e}")
                        
        avg_l2 = np.mean(scenario_errors)
        
        all_results.append({
            "Model": "PINN",
            "Scenario": sc["name"],
            "Mean_error": avg_l2,
            "Std_Loss": np.std(scenario_errors),
            "Resolution": sc["res"],
            "Amount of samples": num_test_samples
        })
        log.info(f"Finished {sc['name']} | Relative L2: {avg_l2:.4f} ± {np.std(scenario_errors):.4f}")

    df = pd.DataFrame(all_results)
    df.to_csv("pinn_model_comparison_results.csv", index=False)
    log.success("PINN Evaluation Complete! Metrics saved cleanly.")

if __name__ == "__main__":
    darcy_pinn_trainer()