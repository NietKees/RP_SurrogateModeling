import torch

import physicsnemo
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.metrics.general.mse import mse
from physicsnemo.models.fno.fno import FNO
from data_utils import get_darcy_loader
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

normaliser = { # Dictionary with mean and std of the permeability and darcy fields
    "permeability": (1.25, 0.75), 
    "darcy": (4.52e-2, 2.79e-2),
}

# Define paths
TRAIN_FILE = "/scratch/jhmtimmermans/benchmarks/PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5"

# Initialize Loader
# dataloader = get_darcy_loader(TRAIN_FILE, batch_size=32, num_workers=4)


dataloader = Darcy2D(
    resolution=64, batch_size=32, nr_permeability_freq=5, normaliser=normaliser
)
model = FNO(
    in_channels=1,
    out_channels=1,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=2,
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=12,
    padding=5,
).to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: 0.85**step
)

# run for 20 iterations
dataloader = iter(dataloader)
for i in range(100):
    batch = next(dataloader)
    truth = batch["darcy"]
    pred = model(batch["permeability"])
    loss = mse(pred, truth)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if i % 10 == 0:
        torch.save(model.state_dict(),'checkpoints/' ,f"checkpoint_{i}.pt")
    print(f"Iteration: {i}. Loss: {loss.detach().cpu().numpy()}")

torch.save(model.state_dict(), "darcy_fno.pt")
