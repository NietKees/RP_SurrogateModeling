import torch
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


from validator import GridValidator
from physicsnemo.utils.logging import LaunchLogger
import torch.nn.functional as F
from Burgers.generator import get_burgers_batch

def get_darcy_setup(cfg=None, resolution=64, batch_size=32):
    """
    Centralized data and validation setup.
    If cfg is provided, it uses values from the YAML. 
    Otherwise, it uses the provided function arguments and hardcoded defaults.
    """
    # 1. Handle Normalization Logic
    if cfg and hasattr(cfg, 'normaliser'):
        normaliser = {
            "permeability": (cfg.normaliser.permeability.mean, cfg.normaliser.permeability.std_dev),
            "darcy": (cfg.normaliser.darcy.mean, cfg.normaliser.darcy.std_dev),
        }
    else:
        # Fallback to your centralized defaults
        normaliser = {
            "permeability": (1.25, 0.75), 
            "darcy": (4.52e-2, 2.79e-2),
        }

    # 2. Extract resolution and batch size
    res = cfg.training.resolution if cfg else resolution
    bs = cfg.training.batch_size if cfg else batch_size

    # 3. Create Dataloader
    dataloader = Darcy2D(
        resolution=res, 
        batch_size=bs, 
        nr_permeability_freq=5, 
        normaliser=normaliser
    )

    # 4. Create Validator
    # outsourced here so all scripts use the same MSE logic
    # validator = GridValidator(loss_fun=MSELoss(reduction="mean"))
    validator = GridValidator(
        loss_fun=RelativeL2Loss()
    )
    return dataloader, validator

def get_burgers_setup(cfg):
    """Factory function creating the DataLoader and GridValidator for Burgers 1D+Time."""
    
    # Extract structural configuration properties from Hydra DictConfig safely
    nx = getattr(cfg.training, "nx", 256)
    nt = getattr(cfg.training, "nt", 100)
    nu = getattr(cfg.physics, "nu", 0.05)
    tmax = getattr(cfg.physics, "tmax", 1.0)
    
    batch_size = cfg.training.batch_size
    num_samples = getattr(cfg.training, "pseudo_epoch_sample_size", 1000)

    # Instantiate the standard training dataset
    dataset = get_burgers_batch(num_samples, nx, nt, nu, tmax)
    validation_dataset = get_burgers_batch(cfg.validation.sample_size, nx, nt, nu, tmax)

    print(f'Generated a dataset of:')
    print(f'nx={nx}, nt={nt}, nu={nu}, tmax={tmax}, batchsize={batch_size}, num_samples={num_samples}, datasetlength={len(dataset)}, validationlenght={len(validation_dataset)}')
    # Build the DataLoader wrapper
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Initialize your visualization/error grid tracker
    validator = GridValidator(loss_fun=RelativeL2Loss())

    return dataloader, validation_dataloader, validator

def prepare_pinn_data(batch, device):
    """Converts grid data to point-wise data for MLP/PINNs"""
    perm = batch["permeability"].to(device)
    B, _, H, W = perm.shape
    
    x = torch.linspace(0, 1, H, device=device)
    y = torch.linspace(0, 1, W, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    coords = torch.stack([grid_x, grid_y], dim=-1).repeat(B, 1, 1, 1).view(-1, 2)
    a_vals = perm.permute(0, 2, 3, 1).reshape(-1, 1)
    return coords, a_vals


class RelativeL2Loss(torch.nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):

        batch_size = pred.shape[0]

        pred = pred.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)

        diff_norm = torch.norm(pred - target, p=2, dim=1)
        target_norm = torch.norm(target, p=2, dim=1)

        rel_l2 = diff_norm / (target_norm + self.eps)

        return rel_l2.mean()
    
    