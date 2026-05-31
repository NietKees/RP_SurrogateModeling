import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for remote server executions
import matplotlib.pyplot as plt

from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

def denormalize(tensor, mean, std):
    return (tensor * std) + mean

def get_physics_informer(device, equation, method="finite_difference", res=64):
    if(equation == 'darcy'):
        forcing_fn = 1.0 
        equation = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)
    else:
        raise ValueError('Equation not defined')

    eq_name = equation.__class__.__name__.lower()
    out_key = f"{eq_name}_u"
    dx = 1.0 / (res + 1)
    
    return PhysicsInformer(
        required_outputs=[out_key],
        equations=equation,
        grad_method=method,
        device=device,
        fd_dx=dx if method == "finite_difference" else None
    )

def test_limits_and_visualize():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")

    # Explicit normalizer stats provided
    norm_stats = {
        "permeability": (1.25, 0.75),
        "darcy": (0.0452, 0.0279)
    }

    test_cases = [
        {"name": "Standard (Train)", "k": [0.5, 2.0], "freq": 5},
    ]

    for case in test_cases:
        print(f"\n--- Running Case: {case['name']} ---")
        try:
            datapipe = Darcy2D(
                resolution=64,
                batch_size=1,
                min_permeability=case['k'][0],
                max_permeability=case['k'][1],
                nr_permeability_freq=case['freq'],
                normaliser=norm_stats,
                device=device
            )

            batch = next(iter(datapipe))
            k_norm = batch["permeability"]  
            u_norm = batch["darcy"]         

            # 1. DENORMALIZE to original physical scale
            k_phys = denormalize(k_norm, norm_stats["permeability"][0], norm_stats["permeability"][1])
            u_phys = denormalize(u_norm, norm_stats["darcy"][0], norm_stats["darcy"][1])

            # 2. Initialize Informer using FINITE DIFFERENCE 
            informer = get_physics_informer(device, equation='darcy', method="finite_difference", res=64)
            
            # 3. Compute residuals on physical scale (Finite difference doesn't need external coordinates)
            with torch.no_grad():
                residuals = informer.forward({
                    "u": u_phys, 
                    "k": k_phys
                })
            pde_residual_tensor = residuals["diffusion_u"]
            
            # 4. Calculate metrics
            mean_full = torch.mean(torch.abs(pde_residual_tensor)).item()
            # Crop away the outer boundary padding layer where the Warp data loader cut off the edges
            pde_cropped = pde_residual_tensor[..., 3:-3, 3:-3]
            mean_cropped = torch.mean(torch.abs(pde_cropped)).item()

            print(f"  Permeability Range: {case['k']}")
            print(f"  Physical Max Expected: {u_phys.max().item():.4e} | Min: {u_phys.min().item():.4e}")
            print(f"  ---> PHYSICAL MEAN RESIDUAL (FULL):    {mean_full:.6f}")
            print(f"  ---> PHYSICAL MEAN RESIDUAL (CROPPED): {mean_cropped:.6f}")
            
            # 5. VISUALIZATION
            u_phys_2d = u_phys.squeeze().cpu().detach().numpy()
            res_2d = torch.abs(pde_residual_tensor).squeeze().cpu().detach().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left Plot: True physical scale pressure field
            im0 = axes[0].imshow(u_phys_2d, cmap='viridis')
            axes[0].set_title(f"Physical Darcy Pressure (u)\nMax: {u_phys_2d.max().item():.4f}")
            fig.colorbar(im0, ax=axes[0])

            # Right Plot: Exact physical error distribution map
            vmax_val = np.percentile(res_2d, 95) if np.any(res_2d) else 1.0
            im1 = axes[1].imshow(res_2d, cmap='hot', vmin=0, vmax=vmax_val) 
            axes[1].set_title(f"Absolute Physical PDE Residual\nMean Full: {mean_full:.4f} | Cropped: {mean_cropped:.4f}")
            fig.colorbar(im1, ax=axes[1])

            plt.tight_layout()
            
            output_filename = "pde_debug_residual.png"
            plt.savefig(output_filename, dpi=150)
            print(f"\n[SUCCESS] Heatmap plot saved successfully to remote disk as: '{output_filename}'")
            plt.close(fig)

        except Exception as e:
            print(f"  CRITICAL ERROR DURING TEST: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_limits_and_visualize()