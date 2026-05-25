import torch
import os
import hydra
import physicsnemo
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.metrics.general.mse import mse
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physics_utils import get_physics_informer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@hydra.main(version_base="1.3", config_path=".", config_name="Darcy_PINN_config")
def darcy_pinn_trainer(cfg: DictConfig )-> None:
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    print(f"Running on device: {device}")
    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    darcy_eq = Diffusion(T="u", D=1.0, dim=2, time=False, Q=1)
    # pde_nodes = darcy_eq.make_nodes();
    """
        PhysicsInformer:
    """
    # physicsInformer = PhysicsInformer(
    #     required_outputs=["diffusion_u"],
    #     equations=darcy_eq,
    #     grad_method="autodiff",
    #     device=device
    # )
    physicsInformer = get_physics_informer(device, 'darcy', method="autodiff")

    """
        End of physics informer
    """
    
    model = FullyConnected(
        in_features=3,      # x, y, and a
        out_features=1,     # u (pressure)
        num_layers=6,
        layer_size=512,
        activation_fn="silu" # PhysicsNeMo default
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    

    @StaticCaptureTraining(model=model, optim=optimizer)
    def forward_train(coords, a_vals):
        coords.requires_grad_(True)
        a_vals.requires_grad_(True)

        input_tensor = torch.cat([coords, a_vals], dim=1)
        u = model(input_tensor)


        # residuals = pde_node.evaluate(var_dict)["diffusion_u"]
        residuals = physicsInformer.forward(
            {
                "coordinates": coords,
                "u": u,
                "a": a_vals
            }
        )
        
        # 2. Extract the specific tensor from the dictionary
        pde_residual = residuals["diffusion_u"]
        
        # 3. Calculate MSE (Mean of Squares)
        loss = torch.mean(torch.square(pde_residual))

        return loss

    print("PINN started training")

    normaliser = { # Dictionary with mean and std of the permeability and darcy fields
        "permeability": (1.25, 0.75), 
        "darcy": (4.52e-2, 2.79e-2),
    }
    dataloader = dataloader = Darcy2D(
        resolution=64, batch_size=32, nr_permeability_freq=5, normaliser=normaliser
    )
    dataloader = iter(dataloader)

    # Training
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
        
        # loss = forward_train(coords, a_vals)

        num_samples = 4096
        idx = torch.randperm(coords.size(0), device=device)[:num_samples]
        
        loss = forward_train(coords[idx], a_vals[idx], )
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | PDE Loss: {loss.item():.6e}")

        if epoch % 100 == 0:
            torch.save(model.state_dict(),'checkpoints/' ,f"checkpoint_{epoch}.pt")
                  

    torch.save(model.state_dict(), "darcy_PINN.pt")


if __name__ == "__main__":
    darcy_pinn_trainer()