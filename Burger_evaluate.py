import os
import torch
import hydra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from math import ceil
from omegaconf import DictConfig
from physicsnemo.models.fno import FNO
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from validator import GridValidator
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureEvaluateNoGrad, load_checkpoint
from data_utils import get_darcy_setup, get_burgers_setup
from Burgers.generator import get_burgers_batch
from torch.utils.data import Dataset, DataLoader

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

@hydra.main(version_base="1.3", config_path=".", config_name="Burgers_pipeline_config")
def test_models_ood(cfg: DictConfig):
    # 1. Setup Distributed Environment
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    log = PythonLogger(name="ood_eval")

    # 2. Define Model Paths
    # Ensure these directories exist and contain a 'checkpoint.pt'
    model_configs = {
        "FNO": "./FNO/Burger_checkpoints",
        "PINO": "./PINO/Burger_checkpoints",
        # "PINN": "./PINN/checkpoints"
    }

    # 3. Define Scenarios
    scenarios = [
        {"name": "Standard (Train)", "nu": 0.05, "complexity": 5, "freq_decay": 1.5, "nx": 128, "nt": 100, "tmax": 1.0, "scale": 1.0},
        
        # Test Stiffer Viscosities (Near Shock Formation)
        {"name": "Low Viscosity (Stiff)", "nu": 0.01, "complexity": 5, "freq_decay": 1.5, "nx": 128, "nt": 100, "tmax": 1.0, "scale": 1.0},
        {"name": "Extreme Low Viscosity", "nu": 0.005, "complexity": 5, "freq_decay": 1.5, "nx": 128, "nt": 100, "tmax": 1.0, "scale": 1.0},
        
        # Test Highly Diffusive Fluid Environments
        {"name": "High Viscosity", "nu": 0.20, "complexity": 5, "freq_decay": 1.5, "nx": 128, "nt": 100, "tmax": 1.0, "scale": 1.0},
        
        # Test Amplitude Scaling (OOD Wave Heights)
        {"name": "Larger Amplitude Scale (x1.5)", "nu": 0.05, "complexity": 5, "freq_decay": 1.5, "nx": 128, "nt": 100, "tmax": 1.0, "scale": 1.5},
        {"name": "Smaller Amplitude Scale (x0.5)", "nu": 0.05, "complexity": 5, "freq_decay": 1.5, "nx": 128, "nt": 100, "tmax": 1.0, "scale": 0.5},

        # High-frequency Chaos (Sharp initial jagged wrinkles)
        {"name": "High Complexity Initial", "nu": 0.05, "complexity": 15, "freq_decay": 1.0, "nx": 128, "nt": 100, "tmax": 1.0, "scale": 1.0},
        {"name": "Smooth Initial Condition", "nu": 0.05, "complexity": 2, "freq_decay": 2.5, "nx": 128, "nt": 100, "tmax": 1.0, "scale": 1.0},
        
        # Zero-Shot Super-Resolution (Interpolation)
        {"name": "ZeroShot Spatial SuperRes", "nu": 0.05, "complexity": 5, "freq_decay": 1.5, "nx": 256, "nt": 100, "tmax": 1.0, "scale": 1.0},
        {"name": "ZeroShot Temporal SuperRes (nt=200)", "nu": 0.05, "complexity": 5, "freq_decay": 1.5, "nx": 128, "nt": 200, "tmax": 1.0, "scale": 1.0},
        {"name": "ZeroShot Full SuperRes", "nu": 0.05, "complexity": 5, "freq_decay": 1.5, "nx": 256, "nt": 200, "tmax": 1.0, "scale": 1.0},
        
        # Temporal Extrapolation (Predicting the Future)
        {"name": "Temporal Extrapolation (tmax=2.0)", "nu": 0.05, "complexity": 5, "freq_decay": 1.5, "nx": 128, "nt": 200, "tmax": 2.0, "scale": 1.0},
        {"name": "Long-term Extrapolation (tmax=5.0)", "nu": 0.05, "complexity": 5, "freq_decay": 1.5, "nx": 128, "nt": 200, "tmax": 5.0, "scale": 1.0},
    ]

    # scenarios= [
    #     {"name": "Standard (Train)", "k": [0.5, 2.0], "freq": 5, "res": 128},

    #     {"name": "Smaller range within training range", "k": [0.5, 1.0], "freq": 5, "res": 128},
    #     {"name": "Larger range around training range", "k": [0.0, 3.0], "freq": 5, "res": 128},
    #     {"name": "light underflow", "k": [1e-1, 2.0], "freq": 5, "res": 128},
    #     {"name": "light overflow", "k": [1.0, 5.0], "freq": 5, "res": 128},

    #     {"name": "light below training (Clay)", "k": [1e-1, 0.5], "freq": 5, "res": 128},
    #     {"name": "Medium below training (Clay)", "k": [1e-3, 1e-1], "freq": 5, "res": 128},
    #     {"name": "Extreme Clay", "k": [1e-7, 1e-6], "freq": 5, "res": 128},

    #     {"name": "smaller visosity (0.5)", "k": [0.5 * (1/0.5), 2 * (1/0.5)], "freq": 5, "res": 128},
    #     {"name": "larger visosity (2)", "k": [0.5 * (1/2), 2 * (1/2)], "freq": 5, "res": 128},

    #     {"name": "Medium Flow gravel", "k": [2.0, 5.0], "freq": 5, "res": 128},
    #     {"name": "High Flow Gravel", "k": [5.0, 10.0], "freq": 5, "res": 128},

    #     {"name": "High Complexity", "k": [0.5, 2.0], "freq": 25, "res": 128},
    #     {"name": "ZeroShot_SuperRes_128", "k": [0.5, 2.0], "freq": 5, "res": 256},
    #     {"name": "ZeroShot_SuperRes_256", "k": [0.5, 2.0], "freq": 5, "res": 512}, 
    # ]

    # Persistent normalization from training
    #
    norm = {
        "permeability": (cfg.normaliser.permeability.mean, cfg.normaliser.permeability.std_dev),
        "darcy": (cfg.normaliser.darcy.mean, cfg.normaliser.darcy.std_dev),
    }

    # validator = GridValidator(loss_fun=torch.nn.MSELoss(reduction="mean"))
    _, _, validator = get_burgers_setup(cfg.pino)

    all_results = []

    test_seed = random.randint(0, 2**31 -1)

    # 4. Iterate through Models
    for model_label, ckpt_path in model_configs.items():
        log.info(f"\n\n\n--- Evaluating Model: {model_label} ---")
        
        # Initialize Fresh Model Architecture
        model = FNO(
            in_channels=cfg.pino.arch.in_channels,
            out_channels=cfg.pino.arch.out_features,
            decoder_layers=cfg.pino.arch.decoder_layers,
            decoder_layer_size=cfg.pino.arch.decoder_layer_size,
            dimension=cfg.pino.arch.dimension,
            latent_channels=cfg.pino.arch.latent_channels,
            num_fno_layers=cfg.pino.arch.num_fno_layers,
            num_fno_modes=[16, 12],  # cfg.arch.num_fno_modes,
            padding=cfg.pino.arch.padding,
        ).to(device)

        load_model_weights(model, ckpt_path, device, log)

        # Optimization wrapper
        @StaticCaptureEvaluateNoGrad(
            model=model, logger=log, use_amp=False, use_graphs=False
        )
        def forward_eval(invars, target_shape_t=None):
            if len(invars.shape) == 3 and target_shape_t is not None:
                invars = invars.unsqueeze(-1).repeat(1, 1, 1, target_shape_t)
            return model(invars)
        
        # 5. Iterate through Scenarios
        for sc in scenarios:
            try:

                log.info(f"Running Scenario: {sc['name']} for {model_label}")
                #TODO consider normalizing the input based on the new ranges. Is that then proper out of distribution?
                # Dynamically unpack variables with default fallbacks
                current_tmax = sc.get("tmax", 1.0)
                current_scale = sc.get("scale", 1.0) # Reads amplitude scaling configuration
                num_samples = cfg.get("test_samples", 20)
                
                dataset = get_burgers_batch(
                    num_samples=num_samples,
                    nx=sc["nx"],
                    nt=sc["nt"],
                    nu=sc["nu"],
                    tmax=current_tmax,
                    complexity=sc["complexity"],
                    amp_scale=current_scale, # Injected here
                    freq_decay=sc["freq_decay"],
                    seed=test_seed 
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=32,
                    shuffle=True,
                    drop_last=True
                )
                num_samples = cfg.test_samples 
                total_loss = 0.0

                scenario_errors = []
                
                with LaunchLogger(f"{model_label}_{sc['name']}") as logger:
                    for i, batch in zip(range(num_samples), dataloader):
                        # Inference
                        x_in = batch["v_init"].to(device)
                        y_target = batch["v"].to(device)
                        
                        pred_out = forward_eval(x_in, target_shape_t=y_target.shape[-1])
                        T_steps = y_target.shape[-1]
                        x_in_4d = x_in.unsqueeze(-1).repeat(1, 1, 1, T_steps)
                        
                        val_loss = validator.compare(
                            x_in_4d,
                            y_target,
                            pred_out,
                            i,
                            logger,
                            title=f'Evaluation_Burgers/Burgers_{sc["name"]}_{model_label}'
                        )
                        # --- FIX HERE: Extract the raw python scalar ---
                        if isinstance(val_loss, torch.Tensor):
                            val_loss_scalar = float(val_loss.detach().cpu().item())
                        else:
                            val_loss_scalar = float(val_loss)

                        total_loss += val_loss_scalar
                        scenario_errors.append(val_loss_scalar)
                    
                    avg_l2 = total_loss / num_samples
                    logger.log_epoch({f"{model_label}_{sc['name']}_Avg_L2": avg_l2})
                    
                    all_results.append({
                        "Model": model_label,
                        "Scenario": sc["name"],
                        "Mean_error": np.mean(scenario_errors),
                        "Std_Loss": np.std(scenario_errors),
                        "Spatial_Res": sc["nx"],
                        "Temporal_Res": sc["nt"],
                        "Amount of samples": num_samples
                    })
            except Exception as e:
                print(e)

    

    # 6. Save Summary
    df = pd.DataFrame(all_results)
    df.to_csv("Burgers_model_comparison_results.csv", index=False)
    log.success("Multi-model evaluation complete. Results saved.")

    plot_error_with_uncertainty(all_results)

def plot_error_with_uncertainty(data_list):
    df = pd.DataFrame(data_list)
    plt.figure(figsize=(64, 32))

    # Optional: Enforce a clean, logical order on the X-axis
    scenario_order = [
        "Standard (Train)", 
        "Low Viscosity (Stiff)", 
        "Extreme Low Viscosity", 
        "High Viscosity", 
        "Larger Amplitude Scale (x1.5)",
        "Smaller Amplitude Scale (x0.5)",
        "High Complexity Initial", 
        "Smooth Initial Condition", 
        "ZeroShot Spatial SuperRes",
        "ZeroShot Temporal SuperRes (nt=200)",
        "ZeroShot Full SuperRes",
        "Temporal Extrapolation (tmax=2.0)",
        "Long-term Extrapolation (tmax=5.0)"
    ]
    
    # Cast to categorical to handle plotting order cleanly without a 'Deviation' key
    df['Scenario'] = pd.Categorical(df['Scenario'], categories=scenario_order, ordered=True)
    df = df.sort_values(['Model', 'Scenario'])

    for model_name in df['Model'].unique():
        subset = df[df['Model'] == model_name]
        
        plt.errorbar(
            subset['Scenario'].astype(str), # Convert back to string for clean x-tick rendering
            subset['Mean_error'], 
            yerr=subset['Std_Loss'], 
            label=model_name,
            fmt='-o',        
            capsize=5,       
            elinewidth=1.5,  
            markersize=6
        )
        
    plt.yscale('log')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Relative L2 Error (Mean ± Std Dev)")
    plt.title("Burgers OOD Stability Performance: FNO vs. PINO")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("burgers_error_with_std_dev.png", dpi=200)
    print("Graph successfully saved as burgers_error_with_std_dev.png")
if __name__ == "__main__":
    test_models_ood()