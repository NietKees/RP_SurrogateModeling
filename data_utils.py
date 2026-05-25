import torch
from torch.nn import MSELoss
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from validator import GridValidator
from physicsnemo.utils.logging import LaunchLogger
import torch.nn.functional as F

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

# def run_training_loop(model, dataloader, optimizer, scheduler, cfg, device, phy_informer=None):
    # """
    # If phy_informer is None, this acts as standard FNO training.
    # If phy_informer is provided, it acts as PINO training.
    # """
    # for epoch in range(cfg.max_epochs):
    #     with LaunchLogger("train", epoch=epoch) as log:
    #         for batch in dataloader:
    #             optimizer.zero_grad()
    #             invar, target = batch["permeability"].to(device), batch["darcy"].to(device)

    #             # 1. Prediction & Data Loss
    #             pred = model(invar)
    #             loss_data = F.mse_loss(pred, target)

    #             # 2. Physics Loss (The PINO "Toggle")
    #             if phy_informer is not None:
    #                 # Compute PDE residuals
    #                 res = phy_informer.forward({"u": pred, "k": invar[:, 0:1]})
    #                 pde_res = res["diffusion_u"]
                    
    #                 # Apply the same padding logic you used before
    #                 pde_res = F.pad(pde_res[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
    #                 loss_pde = F.mse_loss(pde_res, torch.zeros_like(pde_res))
                    
    #                 # Total Loss
    #                 loss = loss_data + (cfg.physics_weight * loss_pde)
    #             else:
    #                 loss = loss_data

    #             loss.backward()
    #             optimizer.step()
    #     scheduler.step()
    # return model
"""--------------------------------------------------"""
# import torch
# import h5py
# from torch.utils.data import Dataset, DataLoader

# class PDEBenchDarcyDataset(Dataset):
#     def __init__(self, file_path, transform=None):
#         self.file_path = file_path
#         # Open once to get the length
#         with h5py.File(self.file_path, 'r') as f:
#             self.len = f['tensor'].shape[0]

#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         # We open the file inside __getitem__ to make it compatible 
#         # with DataLoader multiprocessing (num_workers > 0)
#         with h5py.File(self.file_path, 'r') as f:
#             # PDEBench: (batch, x, y, 1) -> FNO: (1, x, y)
#             permeability = torch.from_numpy(f['nu'][idx]).permute(2, 0, 1).float()
#             darcy_field = torch.from_numpy(f['tensor'][idx]).permute(2, 0, 1).float()
            
#         # Normalization (using your values)
#         permeability = (permeability - 1.25) / 0.75
#         darcy_field = (darcy_field - 4.52e-2) / 2.79e-2
        
#         return {"permeability": permeability, "darcy": darcy_field}

# def get_darcy_loader(file_path, batch_size=32, shuffle=True, num_workers=4):
    # dataset = PDEBenchDarcyDataset(file_path)
    # loader = DataLoader(
    #     dataset, 
    #     batch_size=batch_size, 
    #     shuffle=shuffle, 
    #     num_workers=num_workers,
    #     pin_memory=True  # Speeds up host-to-device transfer
    # )
    # return loader