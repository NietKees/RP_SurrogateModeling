import os
import sys
import random
from math import ceil
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from physicsnemo.distributed import DistributedManager
from physics_utils import get_physics_informer
from data_utils import get_darcy_setup
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from physicsnemo.utils import save_checkpoint

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==============================================================================
# FIELD-AWARE P2INN ARCHITECTURE (MEMORY OPTIMIZED)
# ==============================================================================

class DarcyParameterEncoder(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),  # [B, 16, 32, 32]
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),         # [B, 32, 16, 16]
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),         # [B, 64, 8, 8]
            nn.SiLU(),
            nn.Flatten(),                                                  # [B, 4096]
            nn.Linear(4096, 256),
            nn.SiLU(),
            nn.Linear(256, embedding_dim),
            nn.SiLU()
        )

    def forward(self, k_grid):
        return self.net(k_grid)


class DarcyP2INN_Layer(nn.Module):
    def __init__(self, in_features, out_features, embedding_dim=128):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gain_layer = nn.Linear(embedding_dim, out_features)
        self.bias_layer = nn.Linear(embedding_dim, out_features)
        self.activation = nn.SiLU()

    def forward(self, x, embedding):
        x = self.linear(x)
        gamma = self.gain_layer(embedding)
        beta = self.bias_layer(embedding)
        return self.activation(x * gamma + beta)


class ParameterizedDarcyPINN(nn.Module):
    def __init__(self, layers=6, hidden_dim=256, embedding_dim=128):
        super().__init__()
        self.param_encoder = DarcyParameterEncoder(in_channels=1, embedding_dim=embedding_dim)
        
        self.in_layer = nn.Linear(2, hidden_dim)
        self.activation = nn.SiLU()
        
        self.mid_layers = nn.ModuleList([
            DarcyP2INN_Layer(hidden_dim, hidden_dim, embedding_dim) for _ in range(layers - 2)
        ])
        
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, coords, embedding):
        """
        Args:
            coords: Tensor of shape [Total_Points, 2]
            embedding: Tensor of shape [Total_Points, Embedding_Dim] (already expanded)
        """
        x = self.activation(self.in_layer(coords))
        for layer in self.mid_layers:
            x = layer(x, embedding)
        return self.out_layer(x)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# MAIN TRAINING SEQUENCE LOOP
# ==============================================================================

@hydra.main(version_base="1.3", config_path=".", config_name="Darcy_PINN_config")
def train_p2inn(cfg: DictConfig) -> None:
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    set_seed(cfg.get("seed", 42))
    
    log = PythonLogger(name="p2inn_darcy_native")
    log.file_logging()
    LaunchLogger.initialize()

    print(f"--- Starting Native Spatial P2INN Training on {device} ---")
    os.makedirs("./P2INN/checkpoints", exist_ok=True)

    physicsInformer = get_physics_informer(device, 'darcy', method="autodiff")
    dataloader, validator = get_darcy_setup(cfg)
    dataloader_iter = iter(dataloader)

    k_mean, k_std = 1.25, 0.75
    u_mean, u_std = 4.52e-2, 2.79e-2

    model = ParameterizedDarcyPINN(
        layers=cfg.arch.layers, 
        hidden_dim=cfg.arch.layer_size,
        embedding_dim=128
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.gamma)
    
    ckpt_args = {
        "path": "./P2INN/checkpoints",
        "models": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    current_val_error = float('inf')
    epoch = 1

    while epoch < cfg.training.max_epochs + 1 and current_val_error >= cfg.training.target_error_threshold:
        model.train()
        optimizer.zero_grad()
        
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            
        perm = batch["permeability"].to(device) # [B, 1, H, W]
        B, C, H, W = perm.shape

        # 1. Compute latent map embeddings exactly once per unique field
        base_embeddings = model.param_encoder(perm) # [B, 128]

        # 2. Reconstruct spatial coordinate point clouds
        x = torch.linspace(0, 1, H, device=device)
        y = torch.linspace(0, 1, W, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        coords_all = torch.stack([grid_x, grid_y], dim=-1).repeat(B, 1, 1, 1) # [B, H, W, 2]

        # Flatten structural components for routing
        coords_flat = coords_all.view(-1, 2)
        k_flat_norm = perm.permute(0, 2, 3, 1).reshape(-1, 1)
        
        # Expand the pre-computed embeddings across the spatial mesh instances [B * H * W, 128]
        embeddings_flat = base_embeddings.unsqueeze(1).unsqueeze(2).repeat(1, H, W, 1).view(-1, 128)

        # 3. Extract random interior domain collocation points
        num_samples = cfg.training.num_collocation_points
        idx = torch.randperm(coords_flat.size(0), device=device)[:num_samples]
        
        sampled_coords = coords_flat[idx].clone().detach().requires_grad_(True)
        sampled_k_norm = k_flat_norm[idx].clone().detach().requires_grad_(True)
        sampled_embeddings = embeddings_flat[idx]

        # Domain interior prediction evaluation
        u_norm = model(sampled_coords, sampled_embeddings)
        u_phys = u_norm * u_std + u_mean
        k_phys = sampled_k_norm * k_std + k_mean

        residuals = physicsInformer.forward(
            {"coordinates": sampled_coords, "u": u_phys, "k": k_phys}
        )
        pde_key = list(residuals.keys())[0]
        loss_pde = torch.mean(torch.square(residuals[pde_key]))

        # 4. Extract boundary coordinate metrics safely 
        bc_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        bc_mask[0, :] = True; bc_mask[-1, :] = True; bc_mask[:, 0] = True; bc_mask[:, -1] = True
        bc_mask_expanded = bc_mask.repeat(B, 1, 1).view(-1)

        bc_coords = coords_flat[bc_mask_expanded]
        bc_embeddings = embeddings_flat[bc_mask_expanded]

        u_bc_norm = model(bc_coords, bc_embeddings)
        u_bc_phys = u_bc_norm * u_std + u_mean
        loss_bc = torch.mean(torch.square(u_bc_phys))

        # 5. Optimize weights
        loss = (cfg.physics.pde_weight * loss_pde) + (cfg.physics.bc_weight * loss_bc)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6e} | PDE: {loss_pde.item():.4e} | BC: {loss_bc.item():.4e}")

        if epoch % cfg.training.checkpoint_freq == 0:
            save_checkpoint(**ckpt_args, epoch=epoch)

        # ======================================================================
        # VALIDATION MODULE (MEMORY SAFE)
        # ======================================================================
        if epoch % cfg.validation.validation_pseudo_epochs == 0:
            model.eval()
            with LaunchLogger("P2INN_Valid", epoch=epoch) as logger:
                total_error = 0.0
                
                for i, batch in zip(range(validation_iters), dataloader):
                    invar = batch["permeability"].to(device)  # [B, 1, H, W]
                    target = batch["darcy"].to(device)        # [B, 1, H, W]
                    val_B, _, val_H, val_W = invar.shape
                    
                    x_test = torch.linspace(0, 1, val_H, device=device)
                    y_test = torch.linspace(0, 1, val_W, device=device)
                    grid_x_t, grid_y_t = torch.meshgrid(x_test, y_test, indexing='ij')
                    coords_all_t = torch.stack([grid_x_t, grid_y_t], dim=-1).repeat(val_B, 1, 1, 1).view(-1, 2)
                    
                    with torch.no_grad():
                        # Run the encoder once per map in the validation batch
                        val_embeddings_base = model.param_encoder(invar) # [B, 128]
                        val_embeddings_flat = val_embeddings_base.unsqueeze(1).unsqueeze(2).repeat(1, val_H, val_W, 1).view(-1, 128)
                        
                        # Forward pass over the lightweight tensor
                        pred_flat = model(coords_all_t, val_embeddings_flat)
                        pred = pred_flat.view(val_B, val_H, val_W, 1).permute(0, 3, 1, 2)

                    loss_val = validator.compare(
                        invar, target, pred, 
                        step=i, logger=logger, 
                        title=f'validation_{epoch} P2INN'
                    )
                    total_error += float(loss_val.detach().cpu().item())

                current_val_error = total_error / validation_iters
                logger.log_epoch({"relative_l2_physical": current_val_error})
                print(f"--- Epoch {epoch} | Validation Relative L2 Error: {current_val_error:.6f} ---")
        epoch += 1
                  
    save_checkpoint(**ckpt_args, epoch=epoch - 1)
    log.success("P2INN spatial grid training execution sequence completed successfully.")


if __name__ == "__main__":
    train_p2inn()