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
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
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
from math import ceil

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
    # if cfg and hasattr(cfg, 'pino'):
    try:
        cfg = cfg.pino
    except:
        print('no pino found in config')
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    # initialize monitoring
    log = PythonLogger(name="darcy_fno")
    log.file_logging()
    LaunchLogger.initialize()  # PhysicsNeMo launch logger


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
    
    dataloader, validator = get_darcy_setup(cfg)

    ckpt_args = {
        "path": f"./PINO/checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of \
                      batch size: {steps_per_pseudo_epoch * cfg.training.batch_size}"
        )
    if cfg.validation.sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of \
                      batch size: {validation_iters * cfg.training.batch_size}"
        )

    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    # def forward_train(invars, target):
    #     pred = model(invars)
    #     loss = loss_fun(pred, target)
    #     return loss
    def forward_train(invars, target):
        # invars is the permeability field 'k'
        # For physics loss, we need the model to predict 'u'
        B, _, H, W = invars.shape
        x = torch.linspace(0, 1, H, device=device)
        y = torch.linspace(0, 1, W, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        # # Shape: [B, 2, H, W]
        # coords = torch.stack([grid_x, grid_y], dim=-1).repeat(B, 1, 1, 1).view(-1, 2)
        
        # Stack to get 2 channels (x and y), then repeat for Batch size
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        # 2. Enable Gradients
        coords = coords.clone().requires_grad_(True)
        invars.requires_grad_(True)

        # Compute forward pass
        out = model(invars)

        # print(out.shape, invar[:,0:1].shape)
        residuals = phy_informer.forward(
            {
                "u": out,
                "k": invars[:, 0:1],
                "coordinates": coords,
            }
        )

        pde_out_arr = residuals["diffusion_u"]

        pde_out_arr = F.pad(
            pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
        )
        loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))

        # Compute data loss
        loss_data = F.mse_loss(target, out)

        # Compute total loss
        physics_weight = (1.0 / 240.0) * cfg.physics.weight
        loss = loss_data + physics_weight * loss_pde
        # print(f'pde_loss: {loss_pde}, data_loss: {loss_data}')
        # Backward pass and optimizer and learning rate update
        # loss.backward()
        # optimizer.step()
        # logger.log_minibatch(
        #     {"loss_data": loss_data.detach(), "loss_pde": loss_pde.detach()}
        # )
        return loss
    
    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    if loaded_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_epoch + 1}.")

    for pseudo_epoch in range(
        max(1, loaded_epoch + 1), cfg.training.max_pseudo_epochs + 1
    ):
        # Wrap epoch in launch logger for console / MLFlow logs
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
                loss = forward_train(batch["permeability"].to(device), batch["darcy"].to(device))
                logger.log_minibatch({"loss": loss.detach()})
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
        # scheduler.step()

        # save checkpoint
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # validation step
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                for _, batch in zip(range(validation_iters), dataloader):
                    val_loss = validator.compare(
                        batch["permeability"].to(device),
                        batch["darcy"].to(device),
                        forward_eval(batch["permeability"]),
                        pseudo_epoch,
                        logger,
                    )
                    total_loss += val_loss
                logger.log_epoch({"Validation error": total_loss / validation_iters})

        # update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()
        if pseudo_epoch % 10 == 0:
            save_path = os.path.join('checkpoints', f"fno_checkpoint_{pseudo_epoch}.pt")
            torch.save(model.state_dict(), save_path)

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")


if __name__ == "__main__":
    train_pino()
