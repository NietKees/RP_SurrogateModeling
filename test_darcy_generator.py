import torch
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
import numpy as np

def test_limits():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")

    # Define the "Stress Scenarios"
    # Note: We keep batch_size=1 to isolate individual failures
    test_cases = [
        {"name": "Standard (Train)", "k": [0.5, 2.0], "freq": 5},
        {"name": "Medium Clay", "k": [1e-3, 1e-2], "freq": 5},
        {"name": "Extreme Clay", "k": [1e-7, 1e-6], "freq": 5},
        {"name": "High Flow Gravel", "k": [5.0, 10.0], "freq": 5},
        {"name": "High Complexity", "k": [0.5, 2.0], "freq": 25},
    ]

    for case in test_cases:
        print(f"\n--- Running Case: {case['name']} ---")
        try:
            
            # We initialize WITHOUT a normaliser to see the raw physical values
            datapipe = Darcy2D(
                resolution=64,
                batch_size=1,
                min_permeability=case['k'][0],
                max_permeability=case['k'][1],
                nr_permeability_freq=case['freq'],
                nr_multigrids=2,
                max_iterations=100000,
                device=device
            )

            # Generate one batch
            batch = next(iter(datapipe))
            k = batch["permeability"]
            u = batch["darcy"]

            # Calculate stats
            u_max = u.max().item()
            u_min = u.min().item()
            u_nan = torch.isnan(u).sum().item()

            print(f"  Permeability Range: {case['k']}")
            print(f"  Target Max: {u_max:.4e}")
            print(f"  Target Min: {u_min:.4e}")
            
            if u_nan > 0:
                print(f"  RESULT: FAILED ({u_nan} NaNs detected)")
            elif u_max > 1e10:
                print(f"  RESULT: EXPLODED (Values too high for standard visualization)")
            else:
                print(f"  RESULT: SUCCESS")

        except Exception as e:
            print(f"  CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_limits()