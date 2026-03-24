import os
import time
from typing import Optional, Union

import pytorch_lightning as pl
import torch
import json
import numpy as np
from ase.io import write
from torch_geometric.data import Batch

from tspath.alignment import get_rmsd_batched_scatter
from tspath.analysis import get_validity, pca_plot, evaluate_covmat, print_covmat_results
from tspath.generative import (
    DriftingField,
    EquivariantDriftingField,
    GaussianSampler,
    HarmonicSampler,
)
from tspath.model import DistanceEmbedder, GaussianMomentEmbedder
from tspath.utils import (
    batch_inputs_to_atoms,
)

__all__ = ["DriftingMolecules", "Drifting"]

class DriftingMolecules(pl.LightningModule):
    def __init__(
        self,
        model,
        n_neg_per_pos: int,
        drifting_field: Union[DriftingField, EquivariantDriftingField],
        prior_sampler: Union[GaussianSampler, HarmonicSampler] = GaussianSampler(),
        embedder: Union[None, DistanceEmbedder, GaussianMomentEmbedder] = None,
        only_pos_drift: bool = False,
        sample_every_epoch: int = 50,
        identifier: str = "smiles",
        save_folder: Optional[str] = "samples",
        **kwargs,
    ):
        """
        Initialize the Drifting model.

        Args:
            model:
                The neural network model that predicts the flow field.
            n_neg_per_pos:
                The number of negative samples to generate per positive sample.
            drifting_field:
                The drifting field module that computes the drift based on the model's output.
            prior_sampler:
                The sampler to use for generating negative samples from the prior distribution.
            embedder:
                An optional embedder to compute the drift loss in latent space.
            only_pos_drift:
                If True, only use the positive drift (attraction) for computing the
                drifted position and loss. If False, use the full drift (attraction + repulsion).
            sample_every_epoch:
                How often (in epochs) to generate and visualize samples during training.
            identifier: str
                The key in the batch data to use as identifier for saving the samples
                (e.g., "smiles", "reaction_id", etc.)
            save_folder: str
                The folder where to save generated samples and visualizations.
            **kwargs:
                Additional hyperparameters to save.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "drifting_field", "embedder"])
        self.model = model
        self.n_neg_per_pos = n_neg_per_pos
        self.drifting_field = drifting_field
        self.embedder = embedder
        self.prior_sampler = prior_sampler
        self.only_pos_drift = only_pos_drift
        self.sample_every_epoch = sample_every_epoch
        self.identifier = identifier
        self.save_folder = save_folder

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())
        if self.hparams.get("scheduler") is not None:
            scheduler = self.hparams.scheduler(optimizer)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer

    def sample_negative_batch(self, batch, n_neg_per_pos):
        """
        Create a new batch by repeating each graph in the input batch n_neg_per_pos
        times and sampling from the prior for each graph.
        """

        # Repeat each graph in the batch n_neg_per_pos times to create a new batch for sampling
        data_list = batch.to_data_list()
        repeated_list = [data for data in data_list for _ in range(n_neg_per_pos)]
        batch_negative = Batch.from_data_list(repeated_list)

        # Sample from the prior 
        z = self.prior_sampler.sample(
            size=(batch_negative.num_nodes, 3),
            edge_index=batch_negative.bonded_edge_index,
            batch=batch_negative.batch,
            smiles=batch_negative.smiles,
        )
        batch_negative.pos = z
        return batch_negative

    def _compute_drift_coordinate_space(self, x, batch_pos, batch_neg, step):
        """Compute the drift in coordinate space and the corresponding loss."""
        # We got n_pos graphs, where each have n_conformers
        batch_sizes = batch_pos.num_atoms * batch_pos.num_conformers
        conformer_batch = torch.arange(
            batch_pos.num_graphs, device=x.device
        ).repeat_interleave(batch_sizes)
        y_pos = batch_pos.pos.clone()
        z_pos = batch_pos.x.clone()

        # Per class compute the drift seperately
        V_total = torch.zeros_like(x)
        V_pos_total = torch.zeros_like(x)
        for i in range(batch_pos.num_graphs):
            # positions have shape n_conformers*n_atoms, 3
            mask_pos = conformer_batch == i
            y_i_pos = y_pos[mask_pos]

            # atomic numbers have shape n_atoms
            mask_pos = batch_pos.batch == i
            z_i_pos = z_pos[mask_pos].repeat(batch_pos.num_conformers[i])

            # search negative samples corresponding to the current positive sample
            mask_neg = torch.isin(
                batch_neg.batch,
                torch.arange(
                    i * self.n_neg_per_pos,
                    (i + 1) * self.n_neg_per_pos,
                    device=batch_neg.batch.device,
                ),
            )
            x_i = x[mask_neg]
            z_i_neg = batch_neg.x[mask_neg]

            # Compute the drift
            V, V_pos, V_neg, *_ = self.drifting_field(
                x_i.detach(),  # avoid unnecessary gradient tracking
                y_i_pos,
                x_i.detach(),  # avoid unnecessary gradient tracking
                batch_pos.num_atoms[i],
                atomic_numbers_pos=z_i_pos,
                atomic_numbers_neg=z_i_neg,
            )
            V_total[mask_neg] = V
            V_pos_total[mask_neg] = V_pos

        # In case only attraction
        if self.only_pos_drift:
            x_drifted = (x + V_pos_total).detach()
        else:
            x_drifted = (x + V_total).detach()

        # Compute RMSD loss
        loss = get_rmsd_batched_scatter(x, x_drifted, batch_neg.batch).mean()

        # Log metrics
        metrics = {"loss": loss}
        for metric_name, metric in metrics.items():
            self.log(
                f"{step}/{metric_name}",
                metric,
                on_step=(step == "train"),
                on_epoch=(step != "train"),
                prog_bar=False,
                batch_size=batch_neg.num_graphs,
            )
        return loss

    def _compute_drift_embedded_space(self, x, batch_pos, batch_neg, step):
        """Compute the drift in embedded space and the corresponding loss."""
        # We got n_pos graphs, where each have n_conformers
        batch_sizes = batch_pos.num_atoms * batch_pos.num_conformers
        conformer_batch = torch.arange(
            batch_pos.num_graphs, device=x.device
        ).repeat_interleave(batch_sizes)
        y_pos = batch_pos.pos.clone()
        z_pos = batch_pos.x.clone()

        # Per class compute the drift seperately
        loss = 0.0
        for i in range(batch_pos.num_graphs):
            # positions have shape n_conformers*n_atoms, 3
            mask_pos = conformer_batch == i
            y_i_pos = y_pos[mask_pos]

            # atomic numbers have shape n_atoms
            mask_pos = batch_pos.batch == i
            z_i_pos = z_pos[mask_pos].repeat(batch_pos.num_conformers[i])

            # Embedding of y samples
            batch_i_pos = torch.arange(
                batch_pos.num_conformers[i], device=x.device
            ).repeat_interleave(batch_pos.num_atoms[i])
            with torch.no_grad():
                y_i_pos_embedded = self.embedder(
                    positions=y_i_pos, Z=z_i_pos, batch=batch_i_pos
                )

            # search negative samples corresponding to the current positive sample
            mask_neg = torch.isin(
                batch_neg.batch,
                torch.arange(
                    i * self.n_neg_per_pos,
                    (i + 1) * self.n_neg_per_pos,
                    device=x.device,
                ),
            )
            x_i = x[mask_neg]
            z_i_neg = batch_neg.x[mask_neg]

            # Embedding of x samples
            batch_i_neg = torch.arange(
                self.n_neg_per_pos, device=x.device
            ).repeat_interleave(batch_pos.num_atoms[i])
            x_i_embedded = self.embedder(positions=x_i, Z=z_i_neg, batch=batch_i_neg)

            # Call the drift
            V, V_pos, V_neg, *_ = self.drifting_field(
                x_i_embedded.detach(),
                y_i_pos_embedded,
                x_i_embedded.detach(),
            )

            if self.only_pos_drift:
                x_i_drifted = (x_i_embedded + V_pos).detach()
            else:
                x_i_drifted = (x_i_embedded + V).detach()

            loss = loss + torch.nn.functional.mse_loss(x_i_embedded, x_i_drifted)

        # Log metrics
        metrics = {"loss": loss}
        for metric_name, metric in metrics.items():
            self.log(
                f"{step}/{metric_name}",
                metric,
                on_step=(step == "train"),
                on_epoch=(step != "train"),
                prog_bar=False,
                batch_size=batch_neg.num_graphs,
            )
        return loss

    def _step(self, batch_pos, step):

        # Sample n_neg priors per graph
        batch_neg = self.sample_negative_batch(
            batch_pos, n_neg_per_pos=self.n_neg_per_pos
        )

        # Predict using the model
        x = self.model(batch_neg)

        # Per Class compute the drift seperately
        if self.embedder is not None:
            loss = self._compute_drift_embedded_space(x, batch_pos, batch_neg, step)
        else:
            loss = self._compute_drift_coordinate_space(x, batch_pos, batch_neg, step)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        
        if loss.isnan():
            raise ValueError(
                f"NaN loss encountered at {self.current_epoch}, batch {batch_idx}"
            )
        if (
            (self.current_epoch % self.sample_every_epoch == 0)
            and (batch_idx == 0)
            and (self.current_epoch > 0)
        ):
            if self.save_folder is not None:
                save_folder = f"{self.save_folder}/epoch_{self.current_epoch:05d}/train"
            max_num_conformers = batch.num_conformers.max().item()
            self.sample(
                batch, 
                step="train", 
                save_folder=save_folder, 
                save_pca_plot=True, 
                seed=42, 
                n_neg_per_pos=max_num_conformers*2 # at least having 2*n_conformers
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if (
            (self.current_epoch % self.sample_every_epoch == 0)
            and (batch_idx == 0)
            and (self.current_epoch > 0)
        ):
            if self.save_folder is not None:
                save_folder = f"{self.save_folder}/epoch_{self.current_epoch:05d}/val"
            max_num_conformers = batch.num_conformers.max().item()
            self.sample(
                batch, 
                step="val", 
                save_folder=save_folder, 
                save_pca_plot=True, 
                seed=42, 
                n_neg_per_pos=max_num_conformers*2 # at least having 2*n_conformers
            )
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_pos,
        save_folder=None,
        step=None,
        n_neg_per_pos=None,
        save_pca_plot=False,
        seed=None,
        threshold=0.5,
    ):
        """Generate n_neg_per_pos samples per graph"""
        if seed is not None:
            torch.manual_seed(seed)

        was_training = self.model.training
        self.model.eval()

        start_time = time.time()

        # Sample prior noise
        batch_sampling = self.sample_negative_batch(
            batch_pos, n_neg_per_pos=n_neg_per_pos or self.n_neg_per_pos
        )

        # generate samples
        x = self.model(batch_sampling)

        elapsed_time = time.time() - start_time

        # Convert to ASE Atoms
        batch_sampling.pos_generated = x
        atoms_noise = batch_inputs_to_atoms(
            batch_sampling, pos_key="pos", info_keys=[self.identifier]
        )
        atoms_pred = batch_inputs_to_atoms(
            batch_sampling, pos_key="pos_generated", info_keys=[self.identifier]
        )
        atoms_positive = self._get_pos_atoms(batch_pos)

        # Compute metrics (validity)
        metrics_val = get_validity(atoms_pred)
        results = evaluate_covmat(
            atoms_pred, 
            atoms_positive, 
            thresholds=np.arange(0.05, 3.05, 0.05), 
            num_workers=0, 
            worker_fn_type="rmsd_rdkit_wo_h",
            ratio=2.0, # only keep at most 2*n_conformers predictions per reference
        )
        df, metrics_cov = print_covmat_results(results, threshold=threshold)

        metrics = {**metrics_val, **metrics_cov}
        
        # Save samples
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            write(f"{save_folder}/noise.xyz", atoms_noise)
            write(f"{save_folder}/noise.png", atoms_noise[0])
            write(f"{save_folder}/sample.png", atoms_pred[0])
            for i, atoms in enumerate(atoms_pred):
                sample_folder = (
                    f"{save_folder}/{self.identifier}_{atoms.info[self.identifier]}"
                )
                os.makedirs(sample_folder, exist_ok=True)
                atoms.info["sampling_time"] = elapsed_time / len(atoms_pred)
                write(f"{sample_folder}/sample.xyz", atoms, append=True)
                write(f"{save_folder}/sample_db.xyz", atoms, append=True)

            with open(f"{save_folder}/metrics.json", "w") as f:
                json.dump({"step": self.global_step, **metrics}, f, indent=4)
                
            df.to_csv(f"{save_folder}/covmat_results.csv", index=False)
                
            if save_pca_plot:
                pca_plot(
                    atoms_positive,
                    atoms_pred,
                    embedder=DistanceEmbedder(invariant=True),
                    identifier=self.identifier,
                    save_path=f"{save_folder}/pca_plot.png",
                )

        if step is not None:
            for metric_name, metric in metrics.items():
                self.log(
                    f"{step}/{metric_name}",
                    metric,
                    on_step=(step == "train"),
                    on_epoch=(step != "train"),
                    prog_bar=False,
                    batch_size=batch_sampling.num_graphs,
                )

        if was_training:
            self.model.train()

        return atoms_pred, metrics
    
    def _get_pos_atoms(self, batch_pos):
        """Treating each conformer as a separate graph in the batch"""
        
        batch_pos_ = batch_pos.clone()
        # get positive atoms (consiting of sum n_i_conformers_per_graph)
        z_split = torch.split(batch_pos.x, batch_pos.num_atoms.tolist())
        z_pos = torch.cat(
            [
                z_i.repeat(n_conf_i)
                for z_i, n_conf_i in zip(z_split, batch_pos.num_conformers)
            ]
        )

        # treat each conformer as a separate graph in the batch for evaluation
        offsets = [0] + torch.cumsum(batch_pos.num_conformers, dim=0).tolist()[:-1]
        conformer_batch = torch.cat(
            [
                torch.arange(n_conf_i, device=z_pos.device).repeat_interleave(n_atom_i) + offsets[i]
                for i, (n_conf_i, n_atom_i )in enumerate(zip(batch_pos.num_conformers, batch_pos.num_atoms))
            ]
        )
        
        batch_pos_.batch = conformer_batch
        batch_pos_.x = z_pos

        batch_pos_.smiles = [ 
            smi for smi, n_conf_i in zip(batch_pos_.smiles, batch_pos_.num_conformers) 
            for _ in range(n_conf_i) 
        ]
        atoms_positive = batch_inputs_to_atoms(
            batch_pos_, pos_key="pos", info_keys=[self.identifier]
        )
        return atoms_positive


class Drifting(pl.LightningModule):
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
            fig.savefig(
                f"{outdir}/{step}/epoch_{epoch:05d}.png", dpi=100, bbox_inches="tight"
            )

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
