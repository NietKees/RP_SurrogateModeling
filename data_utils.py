import torch
from torch.nn import MSELoss
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from validator import GridValidator
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
    validator = GridValidator(loss_fun=MSELoss(reduction="mean"))

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