import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from ase.io import write
from torch_scatter import scatter_mean

from tspath.utils import (
    batch_inputs_to_atoms,
    sample_noise_like,
    get_shortest_path_fast_batched_x_1,
    get_rmsd_batch,
    get_rmsd_batch_aligned,
)
from tspath.analysis import check_validity


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
    
    @torch.no_grad()
    def sample(
        self, 
        batch, 
        num_steps, 
        save_folder=None, 
        save_trajectory=False, 
        step=None, 
        conditioned=True, 
        guidance_scale=0.0
    ):
        """Generate samples by integrating the learned flow field."""
        
        was_training = self.model.training
        self.model.eval()

        start_time = time.time()
        x1_pred, trajectory = self.generative_scheduler.sample(
            num_steps=num_steps, model=self.model, batch=batch.clone(), 
            conditioned=conditioned, guidance_scale=guidance_scale
        )
        elapsed_time = time.time() - start_time

        # Convert to ASE Atoms
        batch.pos = x1_pred
        atoms_pred = batch_inputs_to_atoms(batch, pos_key="pos")
        
        # Compute metrics
        validity_res = check_validity(atoms_pred)

        stable_ats = np.concatenate(validity_res["stable_atoms"])
        stable_mols = np.array(validity_res["stable_molecules"])
        stable_ats_wo_h = np.concatenate(validity_res["stable_atoms_wo_h"])
        stable_mols_wo_h = np.array(validity_res["stable_molecules_wo_h"])
        connected = np.array(validity_res["connected"])
        connected_wo_h = np.array(validity_res["connected_wo_h"])

        # infer metrics from validity results
        metrics = {
            "frac_stable_atoms": stable_ats.mean(),
            "frac_stable_molecules": stable_mols.mean(),
            "frac_stable_atoms_wo_h": stable_ats_wo_h.mean(),
            "frac_stable_molecules_wo_h": stable_mols_wo_h.mean(),
            "frac_connected_molecules": connected.mean(),
            "frac_connected_molecules_wo_h": connected_wo_h.mean(),
        }

        # Save samples and trajectory if specified
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            for i, atoms in enumerate(atoms_pred):
                sample_folder = f"{save_folder}/rxn_{atoms.info['reaction_id']}"
                os.makedirs(sample_folder, exist_ok=True)
                # atoms.info["rmsd"] = rmsd[i].item()
                atoms.info["sampling_time"] = elapsed_time / len(atoms_pred)
                atoms.write(f"{sample_folder}/sample.xyz")
                write(f"{save_folder}/sample_db.xyz", atoms, append=True)

            if save_trajectory:
                for step_idx, x_step in enumerate(trajectory):
                    batch.pos_ts = x_step
                    atoms_step = batch_inputs_to_atoms(batch, pos_key="pos")
                    for atoms in atoms_step:
                        sample_folder = f"{save_folder}/rxn_{atoms.info['reaction_id']}"
                        atoms.info["step"] = step_idx
                        write(f"{sample_folder}/traj.xyz", atoms, append=True)

        if step is not None:
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
                
        if was_training:
            self.model.train()
        
        return atoms_pred, metrics



class Drifting(pl.LightningModule):
    def __init__(
        self, 
        model,
        sigma=0.5,
        sample_every_epoch=50, 
        n_sample_steps=10, 
        p_null_mask=0.0,
        guidance_scale=0.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.sigma = sigma
        self.sample_every_epoch = sample_every_epoch
        self.n_sample_steps = n_sample_steps
        self.p_null_mask = p_null_mask
        self.guidance_scale = guidance_scale

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.model.parameters())
        scheduler = self.hparams.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def drifting_field(self, x, y_pos, y_neg, sigma, n_atoms):
        
        def rbf_kernel(x, y, sigma):
            diff = x[:, None, :] - y[None, :, :]
            dist2 = (diff ** 2).mean(dim=-1)
            return torch.exp(-dist2 / (2 * sigma ** 2))
        
        def kernel(x, y, sigma):
            diff = x[:, None, :] - y[None, :, :]
            # dont use the norm but rather the rmsd
            dist = diff.norm(dim=-1) / torch.sqrt(n_atoms)
            return torch.exp(-dist / sigma)
        
        # compute kernel weights per molecule
        x_ = x.view(-1, n_atoms*3)
        y_pos_ = y_pos.view(-1, n_atoms*3)
        y_neg_ = y_neg.view(-1, n_atoms*3)
        
        k_pos = kernel(x_, y_pos_, sigma)
        k_neg = kernel(x_, y_neg_, sigma)
        
        # remove self-repulsion
        k_neg.fill_diagonal_(0.0)
        
        # normalize
        w_pos = k_pos / (k_pos.sum(dim=1, keepdim=True) + 1e-8)
        w_neg = k_neg / (k_neg.sum(dim=1, keepdim=True) + 1e-8)
        
        # compute drift as weighted average of differences
        drift_pos = (w_pos[:, :, None] * (y_pos_[None] - x_[:, None])).sum(dim=1)
        drift_neg = (w_neg[:, :, None] * (y_neg_[None] - x_[:, None])).sum(dim=1)
        
        # reshape back to original shape
        drift_pos = drift_pos.view_as(x)
        drift_neg = drift_neg.view_as(x)
        
        return drift_pos - drift_neg
    
    def _step(self, batch, step):
        # Target data
        y = batch.pos.clone()
        
        # class labels (conformers)
        unique_conformers = batch.formula.unique()
        expanded_formula = batch.formula[batch.batch]
        for conf in unique_conformers:
            mask = expanded_formula == conf
            
            # Get sub-batch indices for the current conformer
            _, sub_batch = torch.unique(batch.batch[mask], return_inverse=True)
            
            # Sample prior noise
            z = sample_noise_like(y[mask], sub_batch)
            batch.pos[mask] = z.clone()
            
            # aligned targets to the noisy input
            y[mask] = get_shortest_path_fast_batched_x_1(z, y[mask], sub_batch)

        # generate samples
        x = self.model(batch)

        # drifting field
        v_total = torch.zeros_like(x)
        for conf in unique_conformers:
            mask = expanded_formula == conf
            mask_i = batch.formula == conf
            
            # Get sub-batch indices for the current conformer
            _, sub_batch = torch.unique(batch.batch[mask], return_inverse=True)
            
            # compute drifting field for each conformer
            v = self.drifting_field(
                x=x[mask],
                y_pos=y[mask],
                y_neg=x.detach()[mask],
                sigma=self.sigma,
                n_atoms=batch.num_atoms[mask_i][0],
            )
            v_total[mask] = v
        
        # stop-gradient target
        x_drift = (x + v_total).detach()
        
        # Compute loss
        loss = ((x - x_drift) ** 2).sum(dim=1).mean()

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
        if (self.current_epoch % self.sample_every_epoch == 0) and (batch_idx == 0) and (self.current_epoch > 0):
            self.sample(
                batch, 
                step="train", 
                guidance_scale=self.guidance_scale
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if (self.current_epoch % self.sample_every_epoch == 0) and (self.current_epoch > 0):
            self.sample(
                batch, 
                step="val", 
                guidance_scale=self.guidance_scale
            )
        return loss

    def sample(self):
        raise NotImplementedError("Sample method not implemented yet.")
    
    @torch.no_grad()
    def sample(
        self, 
        batch, 
        save_folder=None, 
        step=None, 
        conditioned=True, 
        guidance_scale=0.0
    ):
        """Generate samples by integrating the learned flow field."""
        
        was_training = self.model.training
        self.model.eval()

        start_time = time.time()
        
        # Sample prior noise
        z = sample_noise_like(batch.pos, batch.batch)
        batch.pos = z

        # generate samples
        x = self.model(batch)
        
        elapsed_time = time.time() - start_time

        # Convert to ASE Atoms
        batch.pos = x
        atoms_pred = batch_inputs_to_atoms(batch, pos_key="pos")
        
        # Compute metrics
        validity_res = check_validity(atoms_pred)

        stable_ats = np.concatenate(validity_res["stable_atoms"])
        stable_mols = np.array(validity_res["stable_molecules"])
        stable_ats_wo_h = np.concatenate(validity_res["stable_atoms_wo_h"])
        stable_mols_wo_h = np.array(validity_res["stable_molecules_wo_h"])
        connected = np.array(validity_res["connected"])
        connected_wo_h = np.array(validity_res["connected_wo_h"])

        # infer metrics from validity results
        metrics = {
            "frac_stable_atoms": stable_ats.mean(),
            "frac_stable_molecules": stable_mols.mean(),
            "frac_stable_atoms_wo_h": stable_ats_wo_h.mean(),
            "frac_stable_molecules_wo_h": stable_mols_wo_h.mean(),
            "frac_connected_molecules": connected.mean(),
            "frac_connected_molecules_wo_h": connected_wo_h.mean(),
        }

        # Save samples and trajectory if specified
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            for i, atoms in enumerate(atoms_pred):
                sample_folder = f"{save_folder}/rxn_{atoms.info['reaction_id']}"
                os.makedirs(sample_folder, exist_ok=True)
                # atoms.info["rmsd"] = rmsd[i].item()
                atoms.info["sampling_time"] = elapsed_time / len(atoms_pred)
                atoms.write(f"{sample_folder}/sample.xyz")
                write(f"{save_folder}/sample_db.xyz", atoms, append=True)

        if step is not None:
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
                
        if was_training:
            self.model.train()
        
        return atoms_pred, metrics