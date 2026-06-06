# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from math import ceil
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from physicsnemo.distributed import DistributedManager
from hydra.utils import to_absolute_path
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.utils.checkpoint import save_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data_utils import get_darcy_setup
from physics_utils import get_physics_informer


@hydra.main(version_base="1.3", config_path="..", config_name="pipeline_config")
def main(cfg: DictConfig):
    cfg = cfg.pino
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LaunchLogger.initialize()

    # Use Diffusion equation for the Darcy PDE
    forcing_fn = 1.0 * 4.49996e00 * 3.88433e-03  # after scaling
    # forcing_fn = 1.0
    darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)

    # dataset = HDF5MapStyleDataset(
    #     to_absolute_path("./datasets/Darcy_241/train.hdf5"), device=device
    # )
    # validation_dataset = HDF5MapStyleDataset(
    #     to_absolute_path("./datasets/Darcy_241/validation.hdf5"), device=device
    # )

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader, validator = get_darcy_setup(cfg)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = FNO(
        in_channels=cfg.arch.in_channels,
        out_channels=cfg.arch.out_features,
        decoder_layers=cfg.arch.decoder_layers,
        decoder_layer_size=cfg.arch.decoder_layer_size,
        dimension=cfg.arch.dimension,
        latent_channels=cfg.arch.latent_channels,
        num_fno_layers=cfg.arch.num_fno_layers,
        num_fno_modes=cfg.arch.num_fno_modes,
        padding=cfg.arch.padding,
    ).to(device)

    phy_informer = PhysicsInformer(
        required_outputs=["diffusion_u"],
        equations=darcy,
        grad_method="finite_difference",
        device=device,
        fd_dx=1 / 240,  # Unit square with resoultion as 240
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=cfg.training.start_lr,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.gamma)
    # --- EMERGENCY MATHEMATICAL ALIGNMENT CHECK ---
    print("=" * 50)
    print("RUNNING MATHEMATICAL ALIGNMENT DIAGNOSTICS")
    print("=" * 50)
    with torch.no_grad():
        diag_batch = next(iter(dataloader))
        diag_k = diag_batch["permeability"].to(device)[:, 0:1]
        diag_u = diag_batch["darcy"].to(device)
        
        # Test 1: Your current setup
        res1 = phy_informer.forward({"u": diag_u, "k": diag_k})["diffusion_u"]
        print(f"Current Config Loss (fd_dx=1.0, Q=0.01748): {F.mse_loss(res1, torch.zeros_like(res1)).item():.4f}")
        
        # Test 2: Unit Square Scale
        test_informer2 = PhysicsInformer(
            required_outputs=["diffusion_u"], equations=darcy, 
            grad_method="finite_difference", device=device, fd_dx=1/240
        )
        res2 = test_informer2.forward({"u": diag_u, "k": diag_k})["diffusion_u"]
        print(f"Scale Config Loss (fd_dx=1/240, Q=0.01748): {F.mse_loss(res2, torch.zeros_like(res2)).item():.4f}")

        # Test 3: Unit Square Scale + Unit Forcing
        darcy_unit_Q = Diffusion(T="u", time=False, dim=2, D="k", Q=1.0)
        test_informer3 = PhysicsInformer(
            required_outputs=["diffusion_u"], equations=darcy_unit_Q, 
            grad_method="finite_difference", device=device, fd_dx=1/240
        )
        res3 = test_informer3.forward({"u": diag_u, "k": diag_k})["diffusion_u"]
        print(f"Standard Config Loss (fd_dx=1/240, Q=1.0): {F.mse_loss(res3, torch.zeros_like(res3)).item():.4f}")
    print("=" * 50)


    steps_per_pseudo_epoch = 32
    for epoch in range(cfg.training.max_pseudo_epochs):
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=steps_per_pseudo_epoch,
            epoch_alert_freq=10,
        ) as log:
            for _, data in zip(range(steps_per_pseudo_epoch), dataloader):
                optimizer.zero_grad()
                
                invar = data["permeability"].to(device)
                outvar = data["darcy"].to(device) # True ground truth field

                # 1. Compute forward model pass
                out = model(invar[:, 0].unsqueeze(dim=1))

                # 2. Compute Interior PDE Residual
                residuals = phy_informer.forward({"u": out, "k": invar[:, 0:1]})
                pde_out_arr = residuals["diffusion_u"]
                pde_out_arr = F.pad(pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
                loss_pde = F.mse_loss(pde_out_arr, torch.zeros_like(pde_out_arr))

                # 3. FIX: FORCE THE BOUNDARY / CHANNELS (Pure Physics Anchor)
                # Instead of matching the entire image (data loss), we pull ONLY the 
                # outer edge values from outvar to anchor the physics solution.
                mask_boundary = torch.ones_like(out)
                mask_boundary[:, :, 2:-2, 2:-2] = 0.0  # Isolate only the perimeter edges
                
                # Penalize edge variance so the wave/field values enter the grid cleanly
                loss_bc = F.mse_loss(out * mask_boundary, outvar * mask_boundary)

                # 4. Balanced Total Loss Equation
                bc_weight = 50.0  # Keep high to resist zero collapse
                pde_weight = 1 / 240 * cfg.physics.weight
                
                loss = (bc_weight * loss_bc) + (pde_weight * loss_pde)

                # Backward pass
                loss.backward()
                optimizer.step()
                
                log.log_minibatch(
                    {"loss_bc": loss_bc.detach(), "loss_pde": loss_pde.detach()}
                )

            # FIX: Step the learning rate decay ONCE per epoch loop
            scheduler.step()
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            
            
            #
            #
            # validation
            #
            #

            validation_iters = 32
            if epoch % cfg.validation.validation_pseudo_epochs == 0:

                with LaunchLogger("valid", epoch=epoch) as logger:
                    total_l2 = 0.0

                    for i, batch in zip(range(validation_iters), dataloader):
                        invars = batch["permeability"].to(device)
                        target = batch["darcy"].to(device)

                        # pred = forward_eval(invars)
                        pred = model(invars[:, 0].unsqueeze(dim=1))

                        # BUG FIX: Use identical physical spaces inside validator 
                        val_loss = validator.compare(
                            invars,
                            target,
                            pred,
                            i,  # Let GridValidator manage plotting internally via sample rank
                            logger,
                            title=f'PINO_ONY_PHYSICS_{epoch}'
                        )
                        
                        # Safe cast handling for return objects
                        total_l2 += float(val_loss)

                    current_val_error = total_l2 / validation_iters
                    logger.log_epoch({"relative_l2_physical": current_val_error})
                    
                    print(f"--- Epoch {epoch} | Combined GridValidator L2 Error: {current_val_error:.6f} ---")

                    # Early stop check right after validation runs
                    if current_val_error < 0.01:
                        log.success(f"Target metric achieved! Error ({current_val_error:.5f}) < {0.01}")
                        break

                


        with LaunchLogger("valid", epoch=epoch) as log:
            # error = validation_step(model, validation_dataloader, epoch)
            error = validator.compare(invar, outvar, out, step=epoch, logger=log, title=f'PIFNO{epoch}')
            log.log_epoch({"Validation error": error})

        save_checkpoint(
            "./PiFNO/checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
