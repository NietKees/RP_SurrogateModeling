import torch
import os
import hydra
import numpy as np
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

# Optimize CUDA allocation strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def compute_2d_mesh_connectivity(H, W, device):
    """
    Generates a node index array and sequential spatial edge definitions
    for a 2D structured mesh grid of size H x W.
    """
    num_nodes = H * W
    node_ids = torch.arange(num_nodes, device=device).reshape(-1, 1)
    grid_indices = torch.arange(num_nodes, device=device).reshape(H, W)
    
    edge_ids = []
    # Edges along the H-direction (i-index steps)
    if H > 1:
        edges_h = torch.stack([grid_indices[:-1, :].reshape(-1), grid_indices[1:, :].reshape(-1)], dim=1)
        edge_ids.append(edges_h)
    # Edges along the W-direction (j-index steps)
    if W > 1:
        edges_w = torch.stack([grid_indices[:, :-1].reshape(-1), grid_indices[:, 1:].reshape(-1)], dim=1)
        edge_ids.append(edges_w)
        
    edge_ids = torch.cat(edge_ids, dim=0)
    return node_ids, edge_ids

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
    
    print(f"Running on device: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # 2. Physics & Normalization Setup (Using Least Squares over Mesh Domain)
    darcy_eq = Diffusion(T="u", D=1.0, dim=2, time=False, Q=1)
    
    # Adapted configuration to mirror the Least Squares layout
    physicsInformer = PhysicsInformer(
        required_outputs=["diffusion_u"],
        equations=darcy_eq,
        grad_method="least_squares",
        bounds=[1.0, 1.0],               # Normalized spatial bounding box domain constraints [X_max, Y_max]
        device=device,
        compute_connectivity=True        # Signals internal graph assembly routines
    )

    k_mean, k_std = 1.25, 0.75
    u_mean, u_std = 4.52e-2, 2.79e-2
    normaliser = {
        "permeability": (k_mean, k_std), 
        "darcy": (u_mean, u_std),
    }

    # 3. Model Architecture
    model = FullyConnected(
        in_features=3,       # x, y, and a
        out_features=1,      # u (pressure)
        num_layers=6,
        layer_size=512,
        activation_fn="silu" 
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pinn.scheduler.initial_lr)

    # 4. Data Pipeline Setup
    resolution = 64
    dataloader = Darcy2D(
        resolution=resolution, batch_size=32, nr_permeability_freq=5, normaliser=normaliser, device=device
    )
    dataloader_iter = iter(dataloader)

    # Precompute static spatial graph structure topology maps for resolution bounds
    # Since resolution remains fixed, connectivity arrays remain invariant across batches
    node_ids, edge_ids = compute_2d_mesh_connectivity(resolution, resolution, device)

    print("PINN started training...")

    # 5. Main Training Loop
    for epoch in range(1, cfg.pinn.training.max_epochs + 1):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            
        perm = batch["permeability"].to(device)
        B, _, H, W = perm.shape

        # Generate Coordinate Space Planes Pointwise
        x = torch.linspace(0, 1, H, device=device)
        y = torch.linspace(0, 1, W, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        coords = torch.stack([grid_x, grid_y], dim=-1).repeat(B, 1, 1, 1).view(-1, 2)
        a_vals = perm.permute(0, 2, 3, 1).reshape(-1, 1)
        
        # Collocation sampling across flattened array batches
        num_samples = 4096
        idx = torch.randperm(coords.size(0), device=device)[:num_samples]
        
        # ------------------------------------------------------------------
        # Numerical Mesh Optimization Sequence (No Autograd Gradients Needed)
        # ------------------------------------------------------------------
        model.train()
        optimizer.zero_grad()

        # Isolate sample segments (No requires_grad flag adjustments needed!)
        sampled_coords = coords[idx].detach().clone()
        sampled_a_vals_norm = a_vals[idx].detach().clone()

        # Map dynamic node/edge ids aligned with our randomized batch indices 
        # For batch evaluation, indices need to reflect local node assignments
        sampled_node_ids = torch.arange(num_samples, device=device).reshape(-1, 1)
        
        # Forward Evaluation Pass
        input_tensor = torch.cat([sampled_coords, sampled_a_vals_norm], dim=1)
        u_norm = model(input_tensor)

        # De-normalize outputs back to raw physical dimensions
        u_phys = u_norm * u_std + u_mean
        k_phys = sampled_a_vals_norm * k_std + k_mean

        # Compute residuals using structural neighborhood derivatives instead of autodiff tapes
        residuals = physicsInformer.forward(
            {
                "coordinates": sampled_coords,
                "nodes": sampled_node_ids,
                "edges": edge_ids[:num_samples // 2], # Maintain structural neighborhood linkages proportion bounds
                "u": u_phys,
                "k": k_phys 
            }
        )
        
        pde_residual = residuals["diffusion_u"]
        loss = torch.mean(torch.square(pde_residual))

        # Backpropagation
        loss.backward()
        optimizer.step()
        # ------------------------------------------------------------------
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | PDE Loss: {loss.item():.6e}")

        if epoch % 100 == 0:
            ckpt_path = os.path.join("checkpoints", f"checkpoint_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    torch.save(model.state_dict(), "darcy_PINN.pt")
    print("Training Complete! Final model saved to darcy_PINN.pt")

if __name__ == "__main__":
    darcy_pinn_trainer()