import torch
import torch.nn.functional as F
import numpy as np

from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

def test_native_physicsnemo_normalized_pino_case():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Official PhysicsNeMo PINO Reference Test on: {device}\n")

    # 1. Geometry Setup 
    # The official dataset uses 1/N grid spacing for its structural layout
    resolution = 64
    dx = 1.0 / resolution  

    # 2. Dataset Normalization Statistics
    norm_stats = {
        "permeability": (1.25, 0.75),
        "darcy": (0.0452, 0.0279)
    }
    k_mean, k_std = norm_stats["permeability"]
    u_mean, u_std = norm_stats["darcy"]

    # 3. Pull a Ground-Truth Batch from the Darcy2D Pipe
    datapipe = Darcy2D(
        resolution=resolution,
        batch_size=1,
        min_permeability=0.5,
        max_permeability=2.0,
        nr_permeability_freq=5,
        normaliser=norm_stats,
        device=device
    )
    batch = next(iter(datapipe))
    k_norm = batch["permeability"]  # Shape: [1, 1, 64, 64]
    u_norm = batch["darcy"]         # Shape: [1, 1, 64, 64]

    # 4. Reconstruct the Adjusted Forcing Function (Q_scaled)
    # This is the exact step from the reference script where forcing is scaled down:
    # scaled_forcing = physical_forcing * (u_std / k_std)
    scaled_forcing_q = 1.0 * (u_std / k_std)
    print(f"Calculated Adjusted Forcing Function (Q_scaled): {scaled_forcing_q:.6f}")

    # 5. Define the Diffusion Equation inside the Normalized Frame
    darcy_equation = Diffusion(T="u", time=False, dim=2, D="k", Q=scaled_forcing_q)

    # 6. Initialize the Native PhysicsInformer Engine
    phy_informer = PhysicsInformer(
        required_outputs=["diffusion_u"],
        equations=darcy_equation,
        grad_method="finite_difference",
        device=device,
        fd_dx=dx
    )

    # 7. Evaluate the Residual in Normalized Space (No Denormalization)
    with torch.no_grad():
        residuals = phy_informer.forward(
            {
                "u": u_norm,
                "k": k_norm,
            }
        )
        pde_out_arr = residuals["diffusion_u"]

        # Evaluate across the official crop frames to witness the balanced alignment
        print(f"\n=================== OFFICIAL PINO VERIFICATION ===================")
        for crop_margin in [2, 4, 6]:
            # Slice away the boundaries dropped or padded by the physics layers
            cropped_residual = pde_out_arr[..., crop_margin:-crop_margin, crop_margin:-crop_margin]
            
            # PhysicsNeMo utilizes F.l1_loss during its standard backprop training pass
            loss_pde_l1 = F.l1_loss(cropped_residual, torch.zeros_like(cropped_residual)).item()
            loss_pde_mse = F.mse_loss(cropped_residual, torch.zeros_like(cropped_residual)).item()
            
            print(f"Crop Layer: c={crop_margin} | L1 Loss: {loss_pde_l1:.6f} | MSE Loss: {loss_pde_mse:.6f}")
        print(f"==================================================================")

if __name__ == "__main__":
    test_native_physicsnemo_normalized_pino_case()