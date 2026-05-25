import torch
import os
import hydra
import physicsnemo
import sys
from math import ceil
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import (
    StaticCaptureTraining,
    StaticCaptureEvaluateNoGrad,
    load_checkpoint,
    save_checkpoint,
)

from dataclasses import dataclass

# PhysicsNeMo Core Imports
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.metrics.general.mse import mse
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physics_utils import get_physics_informer
from data_utils import get_darcy_setup
import torch
import torch.nn as nn
from physicsnemo.utils.logging import LaunchLogger, PythonLogger


@dataclass
class P2INNMetaData(ModelMetaData):
    name: str = "P2INN"
    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
# 1. Define the P2INN Modular Structure
class P2INN(Module):
    _meta = P2INNMetaData()
    def __init__(self, coord_dim=2, param_dim=1, hidden_dim_c=50, hidden_dim_p=150):
        super().__init__(meta=self._meta)
        # Coordinate Encoder: Processes (x, y)
        self.g_c = FullyConnected(in_features=coord_dim, out_features=hidden_dim_c, 
                                  num_layers=3, layer_size=hidden_dim_c)
        
        # Parameter Encoder: Processes the PDE parameters mu
        # Note: h_param is intentionally high-dimensional (150) to capture 
        # complex characteristics [4].
        self.g_p = FullyConnected(in_features=param_dim, out_features=hidden_dim_p, 
                                  num_layers=4, layer_size=hidden_dim_p)
        
        # Manifold Network: Decodes the concatenated latent space
        self.g_g = FullyConnected(in_features=hidden_dim_c + hidden_dim_p, out_features=1, 
                                  num_layers=5, layer_size=hidden_dim_c)

    def forward(self, coords, mu):
        h_coord = self.g_c(coords)
        h_param = self.g_p(mu)
        # Explicitly concatenate representations [5]
        h_concat = torch.cat([h_coord, h_param], dim=-1)
        return self.g_g(h_concat)

# 2. Updated Trainer Function
# @hydra.main(version_base="1.3", config_path=".", config_name="Darcy_PINN_config")
def train_p2inn(cfg: DictConfig):
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
    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    darcy_eq = Diffusion(T="u", D=1.0, dim=2, time=False, Q=1)
    # pde_nodes = darcy_eq.make_nodes();
    """
        PhysicsInformer:
    """
    physicsInformer = get_physics_informer(device, 'darcy', method="autodiff")

    """
        End of physics informer
    """
    
    # Initialize P2INN instead of standard FullyConnected
    # Assuming 'mu' is the scalar coefficient identifying the permeability scale
    model = P2INN(coord_dim=2, param_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.initial_lr)

    @StaticCaptureTraining(model=model, optim=optimizer)
    def forward_train(coords, mu_vals, a_field_vals):
        coords.requires_grad_(True)
        mu_vals.requires_grad_(True)

        # input_tensor = torch.cat([coords, mu_vals], dim=1)
        # u = model(input_tensor)
        # Forward pass using the separated encoders [2]
        u = model(coords, mu_vals)

        # Residuals computed using point-wise permeability field values 'a'
        residuals = physicsInformer.forward({
            "coordinates": coords,
            "u": u,
            "k": a_field_vals
        })
        return torch.mean(torch.square(residuals["diffusion_u"]))

    # ... [Dataloader setup] ...
    dataloader, validator = get_darcy_setup(cfg)
    dataloader_iter = iter(dataloader)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.training.gamma,
    )
    ckpt_args = {
        "path": "./PINN/checkpoints",
        "models": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

    loaded_epoch = load_checkpoint(device=device, **ckpt_args)

    current_val_error = float('inf')
    validation_iters = ceil(
        cfg.validation.sample_size /
        cfg.training.batch_size
    )
    for epoch in range(1, cfg.training.max_epochs + 1):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        perm = batch["permeability"].to(device)
        truth = batch["darcy"].to(device)
        B, _, H, W = perm.shape
        
        # process for pinn
        x = torch.linspace(0, 1, H, device=device)
        y = torch.linspace(0, 1, W, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        # Flatten everything for the PINN (Point-wise training)
        # coords: [B*H*W, 2], a_vals: [B*H*W, 1]
        coords = torch.stack([grid_x, grid_y], dim=-1).repeat(B, 1, 1, 1).view(-1, 2)
        a_vals = perm.permute(0, 2, 3, 1).reshape(-1, 1)
        
        # KEY CHANGE: Extract the parameter 'mu' characterizing this instance
        # If your generator uses a mean permeability k, use that as mu [6].
        # For simplicity, we'll use the mean of the permeability field in this batch.
        mu_instance = perm.view(B, -1).mean(dim=1, keepdim=True) # [B, 1]
        
        # Expand mu to match collocation points [B*H*W, 1]
        mu_vals = mu_instance.repeat_interleave(H*W, dim=0)
        
        # Sample points and train
        idx = torch.randperm(coords.size(0), device=device)[:4096]
        loss = forward_train(coords[idx], mu_vals[idx], a_vals[idx])
        

        # Generate coordinates and values exactly as you have it...
        # coords = torch.stack([grid_x, grid_y], dim=-1).repeat(B, 1, 1, 1).view(-1, 2)
        # a_vals = perm.permute(0, 2, 3, 1).reshape(-1, 1)
        # mu_instance = perm.view(B, -1).mean(dim=1, keepdim=True)
        # mu_vals = mu_instance.repeat_interleave(H*W, dim=0)
        
# # --- FIXED SAMPLING METHOD ---
#         # 1. Sample indices randomly
#         idx = torch.randperm(coords.size(0), device=device)[:4096]
        
#         # 2. Slice and CLONE to get clean independent tensors outside the tracking function
#         sampled_coords = coords[idx].clone().detach() # Safe to detach out here!
#         sampled_mu = mu_vals[idx].clone().detach()
#         sampled_a = a_vals[idx].clone().detach()
        
#         # 3. Explicitly activate grad tracking BEFORE passing to the framework wrapper
#         sampled_coords.requires_grad_(True)
        
#         # 4. Pass the prepared tensors into the training block
        # loss = forward_train(sampled_coords, sampled_mu, sampled_a)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | PDE Loss: {loss.item():.6e}")

        if epoch % 100 == 0:
            # torch.save(model.state_dict(),'checkpoints/' ,f"checkpoint_{epoch}.pt")
            save_checkpoint(**ckpt_args, epoch=epoch)
            log.success(f"Checkpoint saved at epoch {epoch}")
        
        # ----------------------------------------------------
        # VALIDATION STAGE (FIXED MAPPED DIMENSIONS)
        # ----------------------------------------------------
        if epoch % cfg.validation.validation_pseudo_epochs == 0:
            model.eval()
            val_scenarios = [
                {"name": "Standard_64", "res": 64, "freq": 5},
                {"name": "SuperRes_128", "res": 128, "freq": 5},
                {"name": "HighComplexity", "res": 64, "freq": 25}
            ]

            with LaunchLogger("P2INN_OOD_Valid", epoch=epoch) as logger:
                for scene in val_scenarios:
                    total_scene_error = 0.0
                    scene_loader = get_darcy_setup(res=scene["res"], freq=scene["freq"]) 
                    
                    for i, batch in zip(range(validation_iters), scene_loader):
                        invar = batch["permeability"].to(device)
                        target = batch["darcy"].to(device)
                        B_val, _, H_val, W_val = invar.shape

                        x_v = torch.linspace(0, 1, H_val, device=device)
                        y_v = torch.linspace(0, 1, W_val, device=device)
                        grid_x_v, grid_y_v = torch.meshgrid(x_v, y_v, indexing='ij')
                        coords_val = torch.stack([grid_x_v, grid_y_v], dim=-1).repeat(B_val, 1, 1, 1).view(-1, 2)

                        with torch.no_grad():
                            # FIX 1: Compute instance parameter scalar for validation identically to training
                            mu_val_inst = invar.view(B_val, -1).mean(dim=1, keepdim=True) # [B_val, 1]
                            mu_val_flat = mu_val_inst.repeat_interleave(H_val * W_val, dim=0) # [B_val*H_val*W_val, 1]
                            
                            # FIX 2: Call the fixed model directly using flattened coordinate & param targets
                            pred_flat = model(coords_val, mu_val_flat)
                            pred = pred_flat.view(B_val, H_val, W_val, 1).permute(0, 3, 1, 2)

                        loss_val = validator.compare(
                            invar, target, pred, 
                            step=i, logger=logger, 
                            title=f'{scene["name"]}_Epoch_{epoch}'
                        )
                        total_scene_error += float(loss_val.detach().cpu().item())

                    avg_scene_error = total_scene_error / validation_iters
                    logger.log_epoch({f"rel_l2_{scene['name']}": avg_scene_error})
                    print(f"--- Epoch {epoch} | {scene['name']} Rel L2: {avg_scene_error:.6f} ---")


    save_checkpoint(**ckpt_args, epoch=epoch - 1)
    log.success(f"PINO training complete. Final error achieved: {current_val_error:.5f}")