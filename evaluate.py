import os
import torch
import hydra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from omegaconf import DictConfig
from physicsnemo.models.fno import FNO
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from validator import GridValidator
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureEvaluateNoGrad, load_checkpoint
from data_utils import get_darcy_setup
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
        "FNO": "./FNO/checkpoints",
        "PINO": "./PINO/checkpoints",
        # "PINN": "./PINN/checkpoints"
    }

    # 3. Define Scenarios

    # scenarios = [
    #     {"name": "Standard (Train)", "k": [0.5, 2.0], "freq": 5, "res": 64},
    #     {"name": "light Clay", "k": [1e-2, 1e-1], "freq": 5, "res": 64},
    #     {"name": "Medium Clay", "k": [1e-3, 1e-2], "freq": 5, "res": 64},
    #     {"name": "Extreme Clay", "k": [1e-7, 1e-6], "freq": 5, "res": 64},
    #     {"name": "Medium Flow gravel", "k": [2.0, 5.0], "freq": 5, "res": 64},
    #     {"name": "High Flow Gravel", "k": [5.0, 10.0], "freq": 5, "res": 64},
    #     {"name": "High Complexity", "k": [0.5, 2.0], "freq": 25, "res": 64},
    #     {"name": "ZeroShot_SuperRes_128", "k": [0.5, 2.0], "freq": 5, "res": 128},
    #     {"name": "ZeroShot_SuperRes_256", "k": [0.5, 2.0], "freq": 5, "res": 256},

    #     {"name": "Standard (Train)", "k": [0.5/2, 2.0/2], "freq": 5, "res": 64}, #viscosity of 2

    # ]

    scenarios= [
        # {"name": "Standard (Train)", "k": [0.5, 2.0], "freq": 5, "res": 128},

        # {"name": "Smaller range within training range", "k": [0.5, 1.0], "freq": 5, "res": 128},
        {"name": "Larger range around training range", "k": [0.0, 3.0], "freq": 5, "res": 128},
        # {"name": "light underflow", "k": [1e-1, 2.0], "freq": 5, "res": 128},
        # {"name": "light overflow", "k": [1.0, 5.0], "freq": 5, "res": 128},

        # {"name": "light below training (Clay)", "k": [1e-1, 0.5], "freq": 5, "res": 128},
        # {"name": "Medium below training (Clay)", "k": [1e-3, 1e-1], "freq": 5, "res": 128},
        # {"name": "Extreme Clay", "k": [1e-7, 1e-6], "freq": 5, "res": 128},

        # {"name": "smaller visosity (0.5)", "k": [0.5 * (1/0.5), 2 * (1/0.5)], "freq": 5, "res": 128},
        # {"name": "larger visosity (2)", "k": [0.5 * (1/2), 2 * (1/2)], "freq": 5, "res": 128},

        # {"name": "Medium Flow gravel", "k": [2.0, 5.0], "freq": 5, "res": 128},
        # {"name": "High Flow Gravel", "k": [5.0, 10.0], "freq": 5, "res": 128},

        # {"name": "High Complexity", "k": [0.5, 2.0], "freq": 25, "res": 128},
        # {"name": "ZeroShot_SuperRes_128", "k": [0.5, 2.0], "freq": 5, "res": 256},
        # {"name": "ZeroShot_SuperRes_256", "k": [0.5, 2.0], "freq": 5, "res": 512}, 
    ]

    # Persistent normalization from training
    #
    norm = {
        "permeability": (cfg.normaliser.permeability.mean, cfg.normaliser.permeability.std_dev),
        "darcy": (cfg.normaliser.darcy.mean, cfg.normaliser.darcy.std_dev),
    }

    # validator = GridValidator(loss_fun=torch.nn.MSELoss(reduction="mean"))
    _, validator = get_darcy_setup(cfg)

    all_results = []

    # 4. Iterate through Models
    for model_label, ckpt_path in model_configs.items():
        log.info(f"\n\n\n--- Evaluating Model: {model_label} ---")
        
        # Initialize Fresh Model Architecture
        model = FNO(
            in_channels=cfg.fno.arch.in_channels,
            out_channels=cfg.fno.arch.out_features,
            decoder_layers=cfg.fno.arch.decoder_layers,
            decoder_layer_size=cfg.fno.arch.decoder_layer_size,
            dimension=cfg.fno.arch.dimension,
            latent_channels=cfg.fno.arch.latent_channels,
            num_fno_layers=cfg.fno.arch.num_fno_layers,
            num_fno_modes=cfg.fno.arch.num_fno_modes,
            padding=cfg.fno.arch.padding,
        ).to(device)

        load_model_weights(model, ckpt_path, device, log)

        # Optimization wrapper
        @StaticCaptureEvaluateNoGrad(model=model, logger=log, use_amp=False, use_graphs=False)
        def forward_eval(invars):
            return model(invars)
        
        # 5. Iterate through Scenarios
        for sc in scenarios:
            try:

                log.info(f"Running Scenario: {sc['name']} for {model_label}")
                #TODO consider normalizing the input based on the new ranges. Is that then proper out of distribution?
                k_min, k_max = sc['k'][0], sc['k'][1]
        
                # Simple midpoint/spread calculation
                dynamic_k_mean = (k_max + k_min) / 2
                dynamic_k_std = (k_max - k_min) / 3 # Rough approximation for std dev of a range
                
                # You would also need a heuristic for the 'darcy' (u) field norm, 
                # which is harder because you don't know the output yet.
                # Often, people use the same Darcy norm or scale it linearly with K.
                dynamic_norm = {
                    "permeability": (dynamic_k_mean, dynamic_k_std),
                    "darcy": norm["darcy"] # Keep Darcy fixed or scale it by (dynamic_k_mean / training_k_mean)
                }

                dataloader = Darcy2D(
                    resolution=sc['res'],
                    batch_size=1,
                    min_permeability=sc['k'][0],
                    max_permeability=sc['k'][1],
                    nr_permeability_freq=sc['freq'],
                    normaliser= norm,
                    nr_multigrids=2,
                    max_iterations=100000,
                    # max_iterations=100000,           # Give the solver more time for stiff OOD cases
                    # convergence_threshold=1e-5,       # Slightly loosen the threshold if needed
                    # iterations_per_convergence_check=500,

                    device=device
                )

                num_samples = cfg.test_samples 
                total_loss = 0.0

                scenario_errors = []
                
                with LaunchLogger(f"{model_label}_{sc['name']}") as logger:
                    for i, batch in zip(range(num_samples), dataloader):
                        # Inference
                        invar = batch["permeability"]
                        target = batch["darcy"]
                        pred = forward_eval(invar)

                        # Validation & Plotting
                        # We pass the logger only for the first sample of each scenario to avoid over-plotting
                        val_loss = validator.compare(
                            invar,
                            target,
                            pred,
                            i,
                            logger,
                            title=sc["name"] + "_" + model_label
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
                        # "Avg_L2": avg_l2,
                        "Mean_error": np.mean(scenario_errors),
                        "Std_Loss": np.std(scenario_errors),
                        "Resolution": sc["res"],
                        "Amount of samples": num_samples
                    })
            except Exception as e:
                print(e)

    

    # 6. Save Summary
    df = pd.DataFrame(all_results)
    df.to_csv("model_comparison_results.csv", index=False)
    log.success("Multi-model evaluation complete. Results saved.")

    plot_error_with_uncertainty(all_results)

def plot_error_with_uncertainty(data_list):
    df = pd.DataFrame(data_list)
    plt.figure(figsize=(110, 60))

    # Optional: Enforce a clean, logical order on the X-axis
    scenario_order = [
        "Standard (Train)", 
        "Medium Clay", 
        "Extreme Clay", 
        "High Flow Gravel", 
        "High Complexity", 
        "ZeroShot_SuperRes_128", 
        "ZeroShot_SuperRes_256"
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
    plt.xticks(rotation=35, ha='right')
    plt.ylabel("Relative L2 Error (Mean ± Std Dev)")
    plt.title("OOD Stability Performance: FNO vs. PINO")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig("error_with_std_dev.png")
    print("Graph successfully saved as error_with_std_dev.png")
if __name__ == "__main__":
    test_models_ood()