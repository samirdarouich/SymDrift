import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from ase.io import write
import wandb
from tspath.analysis import check_validity
from tspath.utils import (
    batch_inputs_to_atoms,
    sample_noise_like_2d,
    sample_noise_like,
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
        **kwargs,
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
        if (
            (self.current_epoch % self.sample_every_epoch == 0)
            and (batch_idx == 0)
            and (self.current_epoch > 0)
        ):
            self.sample(batch, step="train", guidance_scale=self.guidance_scale)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if (self.current_epoch % self.sample_every_epoch == 0) and (
            self.current_epoch > 0
        ):
            self.sample(batch, step="val", guidance_scale=self.guidance_scale)
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch,
        save_folder=None,
        save_trajectory=False,
        step=None,
        conditioned=True,
        guidance_scale=0.0,
    ):
        """Generate samples by integrating the learned flow field."""

        was_training = self.model.training
        self.model.eval()

        start_time = time.time()
        x1_pred, trajectory = self.generative_scheduler.sample(
            num_steps=self.n_sample_steps,
            model=self.model,
            batch=batch.clone(),
            conditioned=conditioned,
            guidance_scale=guidance_scale,
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
        drifting_field,
        sample_every_epoch=50,
        p_null_mask=0.0,
        guidance_scale=0.0,
        visualize_type="samples",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.drifting_field = drifting_field
        self.sample_every_epoch = sample_every_epoch
        self.p_null_mask = p_null_mask
        self.guidance_scale = guidance_scale
        self.visualize_type = visualize_type

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())
        if self.hparams.get("scheduler") is not None:
            scheduler = self.hparams.scheduler(optimizer)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            return optimizer

    def _step(self, batch, step):
        # Target data
        y = batch.pos.clone()
        batch.pos_orig = y

        # Sample prior noise
        if self.visualize_type == "2d":
            z = sample_noise_like_2d(y, batch.batch)
        else:
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
            v, *_ = self.drifting_field(
                x=x[mask],
                y_pos=y[mask],
                y_neg=x[mask],
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
        if (
            (self.current_epoch % self.sample_every_epoch == 0)
            and (batch_idx == 0)
            and (self.current_epoch > 0)
        ):
            self.visualize(batch, step="train", outdir="visualizations")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        # if (self.current_epoch % self.sample_every_epoch == 0) and (
        #     self.current_epoch > 0
        # ):
        #     self.sample(batch, step="val", guidance_scale=self.guidance_scale)
        return loss

    @torch.no_grad()
    def sample(
        self, batch, save_folder=None, step=None, conditioned=True, guidance_scale=0.0
    ):
        """Generate samples by integrating the learned flow field."""

        was_training = self.model.training
        self.model.eval()

        start_time = time.time()

        # Sample prior noise
        if self.visualize_type == "2d":
            z = sample_noise_like_2d(batch.pos, batch.batch)
        else:
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
    
    def visualize(self, batch, step, outdir=None):
        pos_dataset = batch.pos_orig.cpu().numpy()
        epoch = self.current_epoch
        atoms_pred, _ = self.sample(batch, step=step)
        
        if outdir is not None:
            os.makedirs(f"{outdir}/{step}", exist_ok=True)

        if self.visualize_type == "samples":
            # Write atoms as png image using ASE's built-in visualization
            for i, atoms in enumerate(atoms_pred):
                sample_path = f"{outdir}/{step}/epoch_{epoch:05d}_sample_{i}.png"
                write(sample_path, atoms, scale=100)
                self.logger.experiment.log(
                    {f"{step}/structure_{i}": wandb.Image(sample_path)}
                )
        elif self.visualize_type == "2d":
            import matplotlib.pyplot as plt
            x_samples = torch.tensor([atoms.get_positions() for atoms in atoms_pred]).view(-1, 3)
            plt.scatter(pos_dataset[:, 0], pos_dataset[:, 1], alpha=0.5, color="gray", label="Dataset")
            plt.scatter(x_samples[:, 0], x_samples[:, 1], alpha=0.5, color="red", label="Samples")
            plt.legend()
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)
            plt.savefig(f"{outdir}/{step}/epoch_{epoch:05d}_samples.png")
            plt.close()


class DriftingDummy(Drifting):

    def _step(self, y, step):

        # Sample prior noise
        z = torch.randn_like(y)

        # generate samples
        x = self.model(z)

        # drifting field
        v_total, *_ = self.drifting_field(
            x=x,
            y_pos=y,
            y_neg=x,
        )

        # stop-gradient target
        x_drifted = (x + v_total).detach()

        # Compute loss
        loss = torch.nn.functional.mse_loss(x, x_drifted)

        # Log metrics
        metrics = {"loss": loss}
        batch_size = y.shape[0]
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

    @torch.no_grad()
    def sample(self, n_samples, y, step=None, **kwargs):
        """Generate samples by integrating the learned flow field."""

        was_training = self.model.training
        self.model.eval()

        start_time = time.time()

        # Sample prior noise (always the same)
        torch.manual_seed(42)
        z = torch.randn(n_samples, *y.shape[1:], device=y.device)

        # generate samples
        x = self.model(z)

        elapsed_time = time.time() - start_time

        if was_training:
            self.model.train()

        return x

    def visualize(self, y, step, n_samples=1000, outdir=None):
        """Plot generated samples vs groundtruth and save/log the figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Get current epoch
        epoch = self.current_epoch
        
        # Generate samples from noise
        x = self.sample(n_samples, y, step=step)

        y_np = y.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(y_np[:, 0], y_np[:, 1], s=3, alpha=0.5, color="steelblue")
        axes[0].set_title("Groundtruth")
        axes[0].set_aspect("equal")

        axes[1].scatter(x_np[:, 0], x_np[:, 1], s=3, alpha=0.5, color="tomato")
        axes[1].set_title(f"Generated (epoch {epoch})")
        axes[1].set_aspect("equal")

        plt.tight_layout()

        if outdir is not None:
            os.makedirs(f"{outdir}/{step}", exist_ok=True)
            fig.savefig(f"{outdir}/{step}/epoch_{epoch:05d}.png", dpi=100, bbox_inches="tight")

        # Log to WandB if available
        if self.logger is not None:
            try:
                import wandb
                self.logger.experiment.log(
                    {f"{outdir}/samples_{step}": wandb.Image(fig)}, epoch=epoch
                )
            except Exception:
                pass

        plt.close(fig)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        if (
            (self.current_epoch % self.sample_every_epoch == 0)
            and (batch_idx == 0)
            and (self.current_epoch > 0)
        ):
            self.visualize(batch, step="train", outdir="visualizations")

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        return loss