import os
import torch
import numpy as np
import hydra
from math import ceil
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))


from omegaconf import DictConfig
from physicsnemo.models.fno import FNO
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
from validator import GridValidator # Assumes validator.py is in your path
from physicsnemo.distributed import DistributedManager
from torch.optim import Adam, lr_scheduler
from physicsnemo.utils import StaticCaptureEvaluateNoGrad, load_checkpoint, save_checkpoint

@hydra.main(version_base="1.3", config_path=".", config_name="pipeline_config")
def test_fno_model(cfg: DictConfig):
    # 1. Setup Device
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    device = dist.device # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    log = PythonLogger(name="darcy_fno")

    model = FNO(
        in_channels=        cfg.fno.arch.in_channels,
        out_channels=       cfg.fno.arch.out_features,
        decoder_layers=     cfg.fno.arch.decoder_layers,
        decoder_layer_size= cfg.fno.arch.decoder_layer_size,
        dimension=          cfg.fno.arch.dimension,
        latent_channels=    cfg.fno.arch.latent_channels,
        num_fno_layers=     cfg.fno.arch.num_fno_layers,
        num_fno_modes=      cfg.fno.arch.num_fno_modes,
        padding=            cfg.fno.arch.padding,
    ).to(device)
    
    
    optimizer = Adam(model.parameters(), lr=cfg.fno.scheduler.initial_lr)

    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.fno.scheduler.decay_rate**step
    )

    ckpt_args = {
        "path": f"./FNO/checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    # 3. Load Checkpoint
    load_checkpoint(device=dist.device, **ckpt_args)
    model.eval()

    print("Model loaded successfully.")

    # 5. Define the Optimized Evaluation Pass
    @StaticCaptureEvaluateNoGrad(model=model, logger=log, use_amp=False, use_graphs=False)
    def forward_eval(invars):
        return model(invars)
    
    # 4. Setup Darcy Generator & Normalization
    # Note: Use the same mean/std you used during training
    norm = {
        "permeability": (1.25, 0.75), 
        "darcy": (0.0452, 0.0279)
    }

    dataloader = Darcy2D(
        resolution=64,
        batch_size=1,
        normaliser=norm,
        device=device
    )

    # datapipe = Darcy2D(
    #     resolution=resolution,
    #     batch_size=num_samples, # used to be 1
    #     min_permeability=k_range[0],
    #     max_permeability=k_range[1],
    #     nr_permeability_freq=nr_freq,
    #     normaliser=self.norm, 
    #     device=self.device
    # )

    # 5. Initialize Validator
    # GridValidator handles L2 error calculation and plotting
    validator = GridValidator(loss_fun=torch.nn.MSELoss(reduction="mean"))
    validation_iters = ceil(cfg.fno.validation.sample_size / cfg.fno.training.batch_size)
    # 6. Run Evaluation Loop
    num_samples = 5
    it = iter(dataloader)
    with LaunchLogger("eval_only") as logger:
        total_loss = 0.0
        for i, batch in zip(range(validation_iters), dataloader):
            # The validator.compare method will automatically generate plots 
            # if a logger is passed and it's the right step.
            val_loss = validator.compare(
                batch["permeability"],
                batch["darcy"],
                forward_eval(batch["permeability"]),
                i, # Use current iteration as step
                logger
            )
            total_loss += val_loss
        
        avg_error = total_loss / validation_iters
        logger.log_epoch({"Final Evaluation L2 Error": avg_error})
        log.success(f"Evaluation Complete. Avg L2 Error: {avg_error:.6f}")
    # LaunchLogger handles the timing and console output formatting
    # with LaunchLogger("Evaluation") as logger:
    #     total_l2 = 0.0

    #     for i in range(num_samples):
    #         batch = next(it)
    #         invar = batch["permeability"]
    #         target = batch["darcy"]

    #         with torch.no_grad():
    #             prediction = model(invar)

    #         # validator.compare returns the L2 error and saves a plot if logger is passed
    #         # We pass the logger only for the first sample to avoid saving 5 identical plots
    #         l2_error = validator.compare(
    #             invar, 
    #             target, 
    #             prediction, 
    #             step=f"sample_{i}", 
    #             logger=logger
    #         )
            
    #         total_l2 += l2_error
    #         print(f"Sample {i} | L2 Error: {l2_error:.6f}")

    #     avg_l2 = total_l2 / num_samples
    #     logger.log_epoch({"Final Average L2": avg_l2})
    #     print(f"\n--- Final Results ---\nAverage L2 Error: {avg_l2:.6f}")

if __name__ == "__main__":
    test_fno_model()