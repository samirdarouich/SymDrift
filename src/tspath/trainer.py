import os
import time

import pytorch_lightning as pl
import torch
from ase.io import write
from torch_scatter import scatter_mean
from tspath.utils import (
    batch_inputs_to_atoms,
    get_rmsd_batch,
    get_rmsd_batch_aligned,
)


class ReactionPath(pl.LightningModule):
    def __init__(
        self, 
        model, 
        generative_scheduler, 
        sample_every_epoch=50, 
        n_sample_steps=10, 
        p_null_mask=0.0,
        guidance_scale=0.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.generative_scheduler = generative_scheduler
        self.sample_every_epoch = sample_every_epoch
        self.n_sample_steps = n_sample_steps
        self.p_null_mask = p_null_mask
        self.guidance_scale = guidance_scale

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.model.parameters())
        scheduler = self.hparams.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def _step(self, batch, step):
        # Use diffusion process to get noisy positions and target noise
        xt, t, eps_target = self.generative_scheduler.sample_t_and_diffuse(
            batch.pos, batch.batch
        )

        batch.pos = xt
        batch.t = t

        # Predict noise
        eps_pred = self.model(batch)

        # Compute loss
        loss = torch.nn.functional.mse_loss(eps_pred, eps_target)

        # Log metrics
        metrics = {"loss": loss}
        batch_size = batch.batch.max().item() + 1
        for metric_name, metric in metrics.items():
            self.log(
                f"{step}/{metric_name}",
                metric,
                on_step=(step == "train"),
                on_epoch=(step != "train"),
                prog_bar=False,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        if (self.current_epoch % self.sample_every_epoch == 0) and batch_idx == 0:
            self.sample(
                batch, 
                self.n_sample_steps, 
                step="train", 
                guidance_scale=self.guidance_scale
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if self.current_epoch % self.sample_every_epoch == 0:
            self.sample(
                batch, 
                self.n_sample_steps, 
                step="val", 
                guidance_scale=self.guidance_scale
            )
        return loss

    def sample(self):
        raise NotImplementedError("Sample method not implemented yet.")
    
    # @torch.no_grad()
    # def sample(
    #     self, 
    #     batch, 
    #     num_steps, 
    #     save_folder=None, 
    #     save_trajectory=False, 
    #     step=None, 
    #     conditioned=True, 
    #     guidance_scale=0.0
    # ):
    #     """Generate samples by integrating the learned flow field."""
        
    #     was_training = self.model.training
    #     self.model.eval()
    
    #     x0 = (batch.pos_r + batch.pos_p) / 2  # Start from the midpoint
    #     x1_target = batch.pos_ts.clone()  # Target is the TS position

    #     start_time = time.time()
    #     x1_pred, trajectory = self.generative_scheduler.sample(
    #         x0=x0, num_steps=num_steps, model=self.model, batch=batch.clone(), 
    #         conditioned=conditioned, guidance_scale=guidance_scale
    #     )
    #     elapsed_time = time.time() - start_time

    #     # Compute final RMSD
    #     rmsd = get_rmsd_batch_aligned(x1_pred, x1_target, batch.batch)

    #     # Convert to ASE Atoms
    #     batch.pos_ts = x1_pred
    #     atoms_pred = batch_inputs_to_atoms(batch, pos_key="pos_ts")

    #     # Save samples and trajectory if specified
    #     if save_folder is not None:
    #         os.makedirs(save_folder, exist_ok=True)
    #         for i, atoms in enumerate(atoms_pred):
    #             sample_folder = f"{save_folder}/rxn_{atoms.info['reaction_id']}"
    #             os.makedirs(sample_folder, exist_ok=True)
    #             atoms.info["rmsd"] = rmsd[i].item()
    #             atoms.info["sampling_time"] = elapsed_time / len(atoms_pred)
    #             atoms.write(f"{sample_folder}/sample.xyz")
    #             write(f"{save_folder}/sample_db.xyz", atoms, append=True)

    #         if save_trajectory:
    #             for step_idx, x_step in enumerate(trajectory):
    #                 batch.pos_ts = x_step
    #                 atoms_step = batch_inputs_to_atoms(batch, pos_key="pos_ts")
    #                 for atoms in atoms_step:
    #                     sample_folder = f"{save_folder}/rxn_{atoms.info['reaction_id']}"
    #                     atoms.info["step"] = step_idx
    #                     write(f"{sample_folder}/traj.xyz", atoms, append=True)

    #     if step is not None:
    #         batch_size = batch.batch.max().item() + 1
    #         metrics = {
    #             "sample_rmsd_mean": rmsd.mean(),
    #             "sample_rmsd_median": rmsd.median(),
    #         }
    #         for metric_name, metric in metrics.items():
    #             self.log(
    #                 f"{step}/{metric_name}",
    #                 metric,
    #                 on_step=(step == "train"),
    #                 on_epoch=(step != "train"),
    #                 prog_bar=False,
    #                 batch_size=batch_size,
    #             )
                
    #     if was_training:
    #         self.model.train()
        
    #     return atoms_pred, rmsd
