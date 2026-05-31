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
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from hydra.utils import to_absolute_path
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from validator import GridValidator
from data_utils import get_darcy_setup
from physics_utils import get_physics_informer

def validation_step(model, dataloader, epoch):
    """Validation Step"""
    model.eval()

    with torch.no_grad():
        loss_epoch = 0
        for data in dataloader:
            invar, outvar, _, _ = data
            out = model(invar[:, 0].unsqueeze(dim=1))

            loss_epoch += F.mse_loss(outvar, out)

        # convert data to numpy
        outvar = outvar.detach().cpu().numpy()
        predvar = out.detach().cpu().numpy()

        # plotting
        fig, ax = plt.subplots(1, 3, figsize=(25, 5))

        d_min = np.min(outvar[0, 0])
        d_max = np.max(outvar[0, 0])

        im = ax[0].imshow(outvar[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(predvar[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(np.abs(predvar[0, 0] - outvar[0, 0]))
        plt.colorbar(im, ax=ax[2])

        ax[0].set_title("True")
        ax[1].set_title("Pred")
        ax[2].set_title("Difference")

        fig.savefig(f"results_{epoch}.png")
        plt.close()
        return loss_epoch / len(dataloader)


@hydra.main(version_base="1.3", config_path="..", config_name="pipeline_config.yaml")
def train_pino(cfg: DictConfig):

    cfg = cfg.pino

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    print(f"Running on device: {device}")
    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LaunchLogger.initialize()

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
    ).to(dist.device)

    phy_informer = get_physics_informer(device, equation=cfg.equation)

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=cfg.training.start_lr,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.gamma)
    
    dataloader, _ = get_darcy_setup(cfg)

    ckpt_args = {
        "path": f"./PINO/checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    for epoch in range(max(1, loaded_epoch + 1), cfg.training.max_pseudo_epochs + 1):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            for data in dataloader:
                optimizer.zero_grad()
                invar = data["permeability"].to(device)
                outvar = data["darcy"].to(device)

                # Compute forward pass
                out = model(invar)

                # print(out.shape, invar[:,0:1].shape)
                residuals = phy_informer.forward(
                    {
                        "u": out,
                        "k": invar[:, 0:1],
                    }
                )
                pde_out_arr = residuals["diffusion_u"]

                pde_out_arr = F.pad(
                    pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                )
                loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))

                # Compute data loss
                loss_data = F.mse_loss(outvar, out)

                # Compute total loss
                loss = loss_data + 1 / 240 * cfg.physics.weight * loss_pde

                # Backward pass and optimizer and learning rate update
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                #
                #   Residual loss on truth
                #
                truth_residuals = phy_informer.forward(
                    {
                        "u": outvar,
                        "k": invar[:, 0:1],
                    }
                )
                t_pde_out_arr = truth_residuals["diffusion_u"]

                t_pde_out_arr = F.pad(
                    t_pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                )
                t_loss_pde = F.l1_loss(t_pde_out_arr, torch.zeros_like(t_pde_out_arr))


                log.log_minibatch(
                    {"loss_data": loss_data.detach(), "loss_pde": loss_pde.detach(), "true_loss": t_loss_pde}
                )

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # with LaunchLogger("valid", epoch=epoch) as log:
            # error = validation_step(model, validator, epoch)
            # log.log_epoch({"Validation error": error})

        # save_checkpoint(
        #     "./checkpoints",
        #     models=model,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     epoch=epoch,
        # )
        # 4. Save using the NVIDIA utility
        if epoch % cfg.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=epoch)


if __name__ == "__main__":
    train_pino()
