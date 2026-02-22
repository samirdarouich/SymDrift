import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from ase.io import write
from tspath.utils import (
    batch_inputs_to_atoms,
    sample_noise_like,
    get_shortest_path_fast_batched_x_1,
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

    @torch.no_grad()
    def sample(
        self, 
        batch, 
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
            num_steps=self.n_sample_steps, model=self.model, batch=batch.clone(), 
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

    def minimal_distance(self, x, y):
        """
        Compute difference between two sets of molecules x and y after optimal alignment.
        x: (N, n_atoms, 3)
        y: (M, n_atoms, 3)
        Returns: diffs matrix of shape (N, M, n_atoms, 3)
        """
        N, n_atoms, _ = x.shape
        M = y.shape[0]
        batch = torch.arange(M, device=x.device).repeat_interleave(n_atoms)  # (M*n_atoms,)
        diffs = []
        for i in range(N):
            x_i = x[i].repeat(M, 1) # (M*n_atoms, 3)
            y_i = y.view(-1, 3)   # (M*n_atoms, 3)
            y_i_aligned = get_shortest_path_fast_batched_x_1(x_i, y_i, batch)
            diff_i = (y_i_aligned-x_i).view(M, n_atoms, 3)
            diffs.append(diff_i)
        diffs = torch.stack(diffs, dim=0) # (N, M, n_atoms, 3)
        return diffs

    def get_weight(self, x, y, sigma, remove_self=False):

            assert x.shape[1] == y.shape[1], "x and y must have the same number of atoms"
            assert x.shape[2] == y.shape[2] == 3, "x and y must have shape (B, n_atoms, 3)"
            n_atoms = x.shape[1]
            
            # Compute aligned differences between each molecule in x and each molecule in y
            diffs = self.minimal_distance(x, y)  # (B, B, n_atoms, 3)
            
            # Treat -1/sigma * rmsd as logits and normalize with softmax
            # this computes the pairwise rmsd between each molecule in x and each
            # molecule in y (x:dim=0, y:dim=1, n_atoms:dim=2, xyz:dim=3)
            rmsd = torch.sqrt((diffs**2).sum(dim=(2,3))/n_atoms)
            logits = -rmsd / sigma
            
            # In case x == y, remove self-repulsion by setting diagonal to -inf before softmax
            if remove_self:
                logits.fill_diagonal_(float('-inf'))
            
            # Normalize weights over y samples for each molecule in x
            return torch.softmax(logits, dim=1), diffs
        
    def drifting_field(self, x, y_pos, y_neg, sigma, n_atoms, normalize_over_x=False):
        
        # reshape to (B, n_atoms, 3) to easily compute drift
        x_ = x.view(-1, n_atoms, 3)
        y_pos_ = y_pos.view(-1, n_atoms, 3)
        y_neg_ = y_neg.view(-1, n_atoms, 3)
        
        # get noramlized kernel (use the original shaped inputs)
        w_pos, diffs_pos = self.get_weight(x_, y_pos_, sigma, remove_self=False)
        w_neg, diffs_neg = self.get_weight(x_, y_neg_, sigma, remove_self=True)
        
        # In addition it can be normalized over x
        if normalize_over_x:
            w_pos = torch.softmax(w_pos.fill_diagonal_(float('-inf')),dim=0) 
            w_neg = torch.softmax(w_neg.fill_diagonal_(float('-inf')),dim=0) 

        # compute drift as weighted average of differences (x is dim=0, y is dim=1).
        # Aim is compute the drift for each molecule in x as a weighted average of the
        # differences to all molecules in y.
        drift_pos = (w_pos[:, :, None, None] * diffs_pos).sum(dim=1)
        drift_neg = (w_neg[:, :, None, None] * diffs_neg).sum(dim=1)
        
        # reshape back to (B*n_atoms, 3)
        drift_pos = drift_pos.view_as(x)
        drift_neg = drift_neg.view_as(x)
        
        return drift_pos - drift_neg

    def _step(self, batch, step):
        # Target data
        y = batch.pos.clone()
        
        # Sample prior noise
        z = sample_noise_like(y, batch.batch)
        batch.pos = z
        
        # generate samples
        x = self.model(batch)

        # drifting field
        v_total = torch.zeros_like(x)
        unique_conformers = batch.formula.unique()
        expanded_formula = batch.formula[batch.batch]
        for conf in unique_conformers:
            mask = expanded_formula == conf
            mask_i = batch.formula == conf

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