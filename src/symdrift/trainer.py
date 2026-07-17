import json
import os
import time
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from ase.io import write
from torch_geometric.data import Batch

from symdrift.analysis import (
    batch_inputs_to_atoms,
    evaluate_covmat,
    get_rmsd_batched_scatter,
    get_validity,
    pca_plot,
    print_covmat_results,
    add_predictions,
)
from symdrift.generative import (
    DriftingField,
    EquivariantDriftingField,
    GaussianSampler,
    HarmonicSampler,
)
from symdrift.model import DistanceEmbedder, GaussianMomentEmbedder
from symdrift.utils import Queue, RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

__all__ = ["DriftingMolecules"]


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
        n_samples: int = 32,
        ratio: int = 2,
        threshold: Optional[float] = 0.5,
        worker_fn_type: str = "rmsd_rdkit_wo_h",
        num_parallel: int = 8,
        grad_norm_max_val: float = 100.0,
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
            n_samples: int
                Number of samples during evaluation in training/validation.
            ratio: int
                Only use n_conformers*ratio generated samples for evaluate coverage and matching.
            threshold: float
                The RMSD threshold to use for evaluating coverage and matching during sampling.
            worker_fn_type: str
                The type of function to use for parallel evaluation of coverage and matching.
            num_parallel: int
                The number of parallel workers/batch size to use for evaluating coverage and matching.
            grad_norm_max_val: float
                The maximum value for the gradient norm when applying adaptive gradient clipping.
            **kwargs:
                Additional hyperparameters to save.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.n_neg_per_pos = n_neg_per_pos
        self.drifting_field = drifting_field
        self.embedder = embedder
        self.prior_sampler = prior_sampler
        self.only_pos_drift = only_pos_drift
        self.sample_every_epoch = sample_every_epoch
        self.identifier = identifier
        self.save_folder = save_folder
        self.n_samples = n_samples
        self.ratio = ratio
        self.threshold = threshold
        self.worker_fn_type = worker_fn_type
        self.num_parallel = num_parallel
        self.grad_norm_max_val = grad_norm_max_val

        # gradient clipping queue
        self.gradnorm_queue = Queue()
        self.gradnorm_queue.add(3000)  # starting value

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())
        if self.hparams.get("scheduler") is not None:
            scheduler = self.hparams.scheduler(optimizer)
            interval = scheduler.interval
            monitor = getattr(scheduler, "monitor", None)
            return [optimizer], [
                {"scheduler": scheduler, "interval": interval, "monitor": monitor}
            ]
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
        batch_negative = Batch.from_data_list(
            repeated_list,
            exclude_keys=[
                # Exlucde all conformer broadcasted properties
                "x_conf",
                "pos",
                "energy",
                "boltzmann_weights",
                "conformer_index",
                "automorphisms",
            ],
        )

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
        # We got n_pos graphs, where each have n_conformers (n_conf_i*n_atom_i)
        y_pos = batch_pos.pos.clone()
        z_pos = batch_pos.x_conf.clone()

        perms_pos = getattr(batch_pos, "automorphisms", None)
        if perms_pos is not None:
            perms_pos = perms_pos.clone()
            
        # Per class compute the drift seperately
        V_total = torch.zeros_like(x)
        V_pos_total = torch.zeros_like(x)
        for i in range(batch_pos.num_graphs):
            # Get all conformers that correspond to the current positive sample
            mask_pos = batch_pos.x_conf_batch == i
            y_i_pos = y_pos[mask_pos]
            z_i_pos = z_pos[mask_pos]
            
            if perms_pos is not None:
                mask_pos_perms = batch_pos.automorphisms_batch == i
                num_perms_i = batch_pos.num_automorphisms[i]
                perms_i_pos = perms_pos[mask_pos_perms].view(num_perms_i, -1)
            else:
                perms_i_pos = None

            # search negative samples corresponding to the current positive sample
            start = i * self.n_neg_per_pos
            end = (i + 1) * self.n_neg_per_pos
            mask_neg = (batch_neg.batch >= start) & (batch_neg.batch < end)
            x_i = x[mask_neg]
            z_i_neg = batch_neg.x[mask_neg]

            # Compute the drift
            with torch.no_grad():
                V, V_pos, V_neg, *_ = self.drifting_field(
                    x_i,
                    y_i_pos,
                    x_i,
                    batch_pos.num_atoms[i],
                    atomic_numbers_pos=z_i_pos,
                    atomic_numbers_neg=z_i_neg,
                    permutations_pos=perms_i_pos,
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
                sync_dist=True,
            )
        return loss

    def _compute_drift_embedded_space(self, x, batch_pos, batch_neg, step):
        """Compute the drift in embedded space and the corresponding loss."""
        # Create batch mask treating each conformer as seperate graph
        batch_mask_pos = batch_pos.conformer_index
        y_pos = batch_pos.pos.clone()
        conformer_offsets = [0] + torch.cumsum(batch_pos.num_conformers, dim=0).tolist()

        # Get orbit ids to define interchangeable pair interactions for the embedder
        orbit_ids_raw = batch_pos.orbit_ids
        device = orbit_ids_raw.device
        
        # Upper-triangle pair count per molecule: n*(n-1)//2
        n_pairs_per_mol = (
            batch_pos.num_atoms * (batch_pos.num_atoms - 1) // 2
        )  # [n_graphs]

        # Expand to per-conformer: repeat each molecule's orbit block
        # n_conformers_i times.
        mol_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            n_pairs_per_mol.cumsum(0),
        ])  # [n_graphs + 1]
        
        orbit_ids_pos = torch.cat([
            orbit_ids_raw[mol_offsets[i] : mol_offsets[i + 1]].repeat(
                batch_pos.num_conformers[i].item()
            )
            for i in range(batch_pos.num_graphs)
        ])  # [sum(n_pairs_i * n_conformers_i)]

        # Negative batch: orbit_ids already replicated n_neg_per_pos times
        # per molecule.
        orbit_ids_neg = batch_neg.orbit_ids

        # Call the embedder for the whole batch. Embedding output is one flatten vector
        # and a mask indicating which embedding belong to which batch element
        y_pos_embedded, mask_pos = self.embedder(
            positions=y_pos,
            batch=batch_mask_pos,
            orbit_ids=orbit_ids_pos,
        )

        x_embedded, mask_x = self.embedder(
            positions=x,
            batch=batch_neg.batch,
            orbit_ids=orbit_ids_neg,
        )

        # Per class compute the drift seperately
        loss = torch.tensor(0.0, device=x.device)
        for i in range(batch_pos.num_graphs):
            # Get all embeddings corresponding to the current positive conformers
            start = conformer_offsets[i]
            end = conformer_offsets[i + 1]
            mask_pos_i = torch.isin(
                mask_pos, torch.arange(start, end, device=mask_pos.device)
            )

            # Reshape to (n_conformers_i, embed_dim_i)
            y_i_pos_embedded = y_pos_embedded[mask_pos_i].view(
                batch_pos.num_conformers[i], -1
            )

            # search negative samples corresponding to the current positive sample
            start = i * self.n_neg_per_pos
            end = (i + 1) * self.n_neg_per_pos
            mask_neg_i = torch.isin(
                mask_x, torch.arange(start, end, device=mask_pos.device)
            )

            # Reshape to (n_neg_per_pos, embed_dim_i)
            x_i_embedded = x_embedded[mask_neg_i].view(self.n_neg_per_pos, -1)

            # Call the drift (stop gradient)
            with torch.no_grad():
                V, V_pos, V_neg, *_ = self.drifting_field(
                    x_i_embedded,
                    y_i_pos_embedded,
                    x_i_embedded,
                )

            if self.only_pos_drift:
                x_i_drifted = (x_i_embedded + V_pos).detach()
            else:
                x_i_drifted = (x_i_embedded + V).detach()

            loss = loss + torch.nn.functional.mse_loss(x_i_embedded, x_i_drifted)

        # normalize loss by the number of graphs in the batch
        loss = loss / batch_pos.num_graphs

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
                sync_dist=True,
            )
        return loss

    def _step(self, batch_pos, step):

        # Sample n_neg priors per graph
        batch_neg = self.sample_negative_batch(
            batch_pos, n_neg_per_pos=self.n_neg_per_pos
        )

        # Generate target samples using the model
        x = self.model(batch_neg)

        if x.isnan().any():
            raise ValueError(
                f"NaN values in model output at epoch {self.current_epoch} and step {step}"
            )

        # Compute the drift seperately per class
        if self.embedder is not None:
            loss = self._compute_drift_embedded_space(x, batch_pos, batch_neg, step)
        else:
            loss = self._compute_drift_coordinate_space(x, batch_pos, batch_neg, step)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")

        if loss.isnan():
            raise ValueError(
                f"NaN loss encountered at epoch {self.current_epoch} and batch {batch_idx}"
            )
        if (
            (self.current_epoch % self.sample_every_epoch == 0)
            and (batch_idx == 0)
            and (self.current_epoch > 0)
        ):
            logger.info(
                f"Sampling at epoch {self.current_epoch} after training step..."
            )
            if self.save_folder is not None:
                save_folder = f"{self.save_folder}/epoch_{self.current_epoch:05d}/train"
            self.sample(
                batch,
                n_samples=self.n_samples,
                step="train",
                save_folder=save_folder,
                save_pca_plot=True,
                seed=42,
                threshold=self.threshold,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if (self.current_epoch % self.sample_every_epoch == 0) and (
            self.current_epoch > 0
        ):
            logger.info(
                f"Sampling at epoch {self.current_epoch} after validation step..."
            )
            if self.save_folder is not None:
                save_folder = f"{self.save_folder}/epoch_{self.current_epoch:05d}/val"
            self.sample(
                batch,
                n_samples=self.n_samples,
                step="val",
                save_folder=save_folder,
                save_pca_plot=batch_idx == 0,  # only save PCA plot for the first batch
                seed=42,
                threshold=self.threshold,
            )
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_pos,
        n_samples,
        num_steps=1,
        save_folder=None,
        step=None,
        save_pca_plot=False,
        seed=None,
        threshold=None,
        **kwargs,
    ):
        """Generate n_neg_per_pos samples per graph"""
        if seed is not None:
            torch.manual_seed(seed)

        was_training = self.model.training
        self.model.eval()

        # Sample prior noise
        batch_sampling = self.sample_negative_batch(batch_pos, n_neg_per_pos=n_samples)
        batch_sampling.pos_noise = batch_sampling.pos.clone()
        
        # generate samples, feeding each prediction back in as the next
        # iteration's input until num_steps (NFE) function evaluations are done
        start_time = time.time()
        for _ in range(num_steps):
            x = self.model(batch_sampling)
            batch_sampling.pos = x
        elapsed_time = time.time() - start_time
        
        # Convert predictions to ASE Atoms 
        batch_sampling.pos_generated = x
        atoms_pred = batch_inputs_to_atoms(
            batch_sampling, pos_key="pos_generated", info_keys=[self.identifier]
        )
        metrics_val = get_validity(atoms_pred)
        
        # Compute metrics (coverage and matching)
        if threshold is not None:
            add_predictions(batch_pos, total_samples=n_samples, positions=x, key="pos_generated")
            results, _ = evaluate_covmat(
                preds=batch_pos.to_data_list(),
                thresholds=np.arange(0.05, 3.05, 0.05),
                num_parallel=self.num_parallel,
                worker_fn_type=self.worker_fn_type,
                ratio=self.ratio,
                identifier=self.identifier,
            )
            df, metrics_cov = print_covmat_results(results, threshold=threshold)
        else:
            df, metrics_cov = None, {}

        metrics = {
            **metrics_val,
            **metrics_cov,
            "sampling_time": elapsed_time / len(atoms_pred),
        }

        # Save samples
        if save_folder is not None and self.is_global_zero():
            os.makedirs(save_folder, exist_ok=True)
            
            # Convert to ASE Atoms
            atoms_noise = batch_inputs_to_atoms(
                batch_sampling, pos_key="pos_noise", info_keys=[self.identifier]
            )
        
            write(f"{save_folder}/noise.xyz", atoms_noise, append=True)
            write(f"{save_folder}/noise.png", atoms_noise[0])
            write(f"{save_folder}/sample.png", atoms_pred[0])
            for atoms in atoms_pred:
                atoms.info["sampling_time"] = elapsed_time / len(atoms_pred)
            write(f"{save_folder}/sample_db.xyz", atoms_pred, append=True)

            with open(f"{save_folder}/metrics.json", "w") as f:
                json.dump({"step": self.global_step, **metrics}, f, indent=4)

            if df is not None:
                df.to_csv(f"{save_folder}/covmat_results.csv", index=False)

            if save_pca_plot:
                atoms_positive = self._get_pos_atoms(batch_pos)
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
                    sync_dist=True,
                )

        if was_training:
            self.model.train()

        return x, atoms_pred, metrics

    def _get_pos_atoms(self, batch_pos):
        """Treating each conformer as a separate graph in the batch"""
        batch_pos_ = batch_pos.clone()
        batch_pos_.batch = batch_pos.conformer_index
        batch_pos_.x = batch_pos.x_conf

        batch_pos_.smiles = [
            smi
            for smi, n_conf_i in zip(batch_pos_.smiles, batch_pos_.num_conformers)
            for _ in range(n_conf_i)
        ]
        atoms_positive = batch_inputs_to_atoms(
            batch_pos_, pos_key="pos", info_keys=[self.identifier]
        )
        return atoms_positive

    def is_global_zero(self):
        return (self._trainer is None) or self.trainer.is_global_zero

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ):
        """Gradient Clipping as done in the official EDM implementation."""

        # In case no max grad norm value is set, use the default Lightning implementation
        if self.grad_norm_max_val is None:
            return super().configure_gradient_clipping(
                optimizer, gradient_clip_val, gradient_clip_algorithm
            )

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = min(
            1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std(),
            self.grad_norm_max_val,  # do not increase the gradient norm beyond 100
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_grad_norm, norm_type=2.0
        )

        if float(grad_norm) > max_grad_norm and grad_norm < self.grad_norm_max_val:
            # only update if grad_norm is not too large
            self.gradnorm_queue.add(max_grad_norm)
        else:
            self.gradnorm_queue.add(grad_norm.cpu().item())

        if float(grad_norm) > max_grad_norm:
            logger.debug(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}"
            )
