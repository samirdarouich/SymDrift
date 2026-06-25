import json
import os
import socket
import uuid
from typing import Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from symdrift.analysis import (
    evaluate_covmat,
    add_predictions,
    print_covmat_results,
)
from symdrift.utils import RankedLogger, log_hyperparameters, print_config

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_new_resolver("replace", lambda s, old, new: s.replace(old, new))

log = RankedLogger(__name__, rank_zero_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="configs", version_base="1.2", config_name="base")
def train(cfg):

    log.info("Running on host: " + str(socket.gethostname()))
    log.info("Starting training for run: {}".format(cfg.run.id))
    if cfg.get("print_config", True):
        fields = (
            "run",
            "generative_model",
            "drifting_field",
            "prior",
            "embedder",
            "model",
            "dataset",
            "trainer",
            "callbacks",
            "loggers",
            "seed",
        )
        print_config(cfg, fields=fields, resolve=False)

    ########## Hyperparameters and settings ##########
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    ########## Datamodule (includes datasets and loaders) ##########
    datamodule = instantiate(cfg.dataset.datamodule)

    ########## Callbacks and Logger ##########
    callbacks, loggers = [], []
    for callback_name, callback in cfg.callbacks.items():
        callbacks.append(instantiate(callback))
    for logger_name, logger in cfg.loggers.items():
        loggers.append(instantiate(logger))

    ########## Train the generative model #############
    generative_process = instantiate(cfg.generative_model)

    # Load pretrained model if specified
    if cfg.generative_model.pretrained is not None:
        log.info(
            f"\n\nLoading pretrained model from <{cfg.generative_model.pretrained}>\n\n"
        )
        pretrained = torch.load(
            cfg.generative_model.pretrained, "cpu", weights_only=False
        )

        if isinstance(pretrained, torch.nn.Module):
            state_dict = pretrained.state_dict()
        elif isinstance(pretrained, dict):
            state_dict = pretrained["state_dict"]
            if "model" in list(state_dict.keys())[0]:
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        generative_process.model.load_state_dict(state_dict)

    generative_trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=os.path.join(cfg.run.id),
        **cfg.trainer,
    )

    log.info("Logging hyperparameters...")
    log_hyperparameters(cfg, generative_trainer)

    log.info("Starting training...")
    generative_trainer.fit(
        model=generative_process,
        datamodule=datamodule,
        ckpt_path=cfg.run.ckpt_path,
        weights_only=False,
    )

    log.info("Training completed.")


@hydra.main(config_path="configs", version_base="1.2", config_name="base")
def sample(cfg):

    log.info("Running on host: " + str(socket.gethostname()))
    log.info("Starting inference for run: {}".format(cfg.run.id))
    if cfg.get("print_config", True):
        fields = (
            "run",
            "generative_model",
            "model",
            "prior",
            "dataset",
            "seed",
        )
        print_config(cfg, fields=fields, resolve=False)

    ########## Hyperparameters and settings ##########
    log.info("Setting random seed to {}".format(cfg.seed))
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    ########## Datamodule (includes datasets and loaders) ##########
    if getattr(cfg.dataset, "test_dataloader", None) is not None:
        log.info("Using seperate test dataset for sampling.")
        dataloader = instantiate(cfg.dataset.test_dataloader)
    else:
        log.info("Using datamodule to load dataset for sampling.")
        datamodule = instantiate(cfg.dataset.datamodule)
        sampling_split = getattr(cfg.dataset, "sampling_split", "test")
        setattr(datamodule, f"{sampling_split}_batch_size", cfg.dataset.test_batch_size)
        datamodule.setup(stage=sampling_split)
        dataloader = getattr(datamodule, f"{sampling_split}_dataloader")()

    generative_process = instantiate(cfg.generative_model)
    log.info("Loading model checkpoint: <{}>".format(cfg.generative_model.pretrained))
    state_dict = torch.load(
        cfg.generative_model.pretrained, weights_only=False, map_location=device
    )["state_dict"]
    generative_process.load_state_dict(state_dict)
    generative_process.to(device)
    generative_process.eval()

    nfe = getattr(cfg.generative_model, "n_sample_steps", 1)
    guidance_scale = getattr(cfg.generative_model, "guidance_scale", 0.0)
    conditioned = getattr(cfg.generative_model, "conditioned", True)
    n_samples = getattr(cfg.generative_model, "n_samples", 1)
    threshold = getattr(cfg.generative_model, "threshold", None)
    ratio = getattr(cfg.generative_model, "ratio", 2.0)
    identifier = getattr(cfg.generative_model, "identifier", "smiles")
    num_parallel = getattr(cfg.generative_model, "num_parallel", 8)
    worker_fn_type = getattr(cfg.generative_model, "worker_fn_type", "rmsd_rdkit_wo_h")
    skip_disconnected = getattr(cfg.generative_model, "skip_disconnected", True)
    job_kwargs = getattr(cfg.generative_model, "job_kwargs", {})
    save_trajectory = getattr(cfg, "save_trajectory", False)
    save_pca_plot = getattr(cfg, "save_pca_plot", False)
    prior_type = generative_process.prior_sampler.type
    base_folder = f"nfe_{nfe}_gs_{guidance_scale}"
    save_folder = base_folder

    if os.path.exists(save_folder):
        log.warning(f"Save folder {save_folder} already exists...")
        i = 1
        while os.path.exists(save_folder):
            save_folder = f"{base_folder}_{i:02d}"
            i += 1
        log.info(f"New save folder: {save_folder}")

    log.info(
        f"Sampling at max {n_samples} time(s) with:\n"
        f"  NFE = {nfe} "
        f"  guidance_scale = {guidance_scale}"
        f"  conditioned = {conditioned}"
        f"  prior_type = {prior_type}"
        f"  N_samples/N_conformers ratio = {ratio}"
    )

    # Fix initial seed for each batch (in case of gaussian prior, this would be
    # the same prior for each batch).
    sample_seed = getattr(cfg, "sample_seed", None)
    if sample_seed is not None:
        log.info(f"Setting sample seed for every batch to {sample_seed}")

    metrics = {}
    atoms_generated = []
    data_generated = []
    for batch in tqdm(dataloader, desc="Evaluating dataset"):
        batch = batch.to(device)

        # In case batch size is bigger than one, it might be that some graphs are
        # oversampled in the evaluation this is corrected by only keeping at most
        # ratio*n_conformers samples per reference graph. But sample at least
        # n_samples times.
        num_conformers = batch.num_conformers.max().item()
        if ratio is not None:
            total_samples = min(int(ratio * num_conformers), n_samples)
        else:
            total_samples = n_samples

        pos_generated = []
        for start in tqdm(
            range(0, total_samples, cfg.dataset.batch_size),
            desc=f"Sampling {total_samples} conformers for batch",
        ):
            cur_n = min(cfg.dataset.batch_size, total_samples - start)
            batch_pos_generated, batch_atoms_generated, batch_metrics = (
                generative_process.sample(
                    batch_pos=batch,
                    num_steps=nfe,
                    n_samples=cur_n,
                    save_folder=save_folder,
                    save_trajectory=save_trajectory,
                    save_pca_plot=save_pca_plot,
                    conditioned=conditioned,
                    guidance_scale=guidance_scale,
                    seed=None if sample_seed is None else sample_seed + start,
                )
            )

            atoms_generated.extend(batch_atoms_generated)
            pos_generated.append(batch_pos_generated)

            for k, v in batch_metrics.items():
                metrics.setdefault(k, []).extend(v if isinstance(v, list) else [v])

        # Add pos_generated to batch and add slicing information in case loop through
        # the dataset is done with batch_size > 1.
        add_predictions(
            batch, 
            total_samples=total_samples, 
            positions=torch.cat(pos_generated, dim=0), 
            key="pos_generated"
        )
        data_generated.extend(batch.to_data_list())

    summary_metrics = {}
    for k, v in metrics.items():
        metrics[k] = torch.tensor(v)
        log.info(
            f"Test {k}: mean: {metrics[k].mean().item():.4f} "
            f"median: {metrics[k].median().item():.4f}"
        )
        summary_metrics[k + "_mean"] = metrics[k].mean().item()
        summary_metrics[k + "_median"] = metrics[k].median().item()

    with open(os.path.join(save_folder, "metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=4)

    # Save generated data as a PyTorch file for later analysis
    torch.save(data_generated, os.path.join(save_folder, "data_generated.pt"))

    # Evaluate coverage and matching for the whole dataset if specified
    if threshold is not None:
        job_kwargs_str = ", "+", ".join(f"{k}={v}" for k, v in job_kwargs.items())
        log.info(
            f"Analysing coverage and matching (threshold: {threshold:.2f}, "
            f"num_parallel: {num_parallel}, worker_fn_type: {worker_fn_type}, "
            f"ratio: {ratio}{job_kwargs_str}):"
        )

        # Evaluate coverage and matching
        results, rmsd_matrix = evaluate_covmat(
            data_generated,
            thresholds=np.arange(0.05, 3.05, 0.05),
            worker_fn_type=worker_fn_type,
            num_parallel=num_parallel,
            ratio=ratio,  # only keep at most ratio*n_conformers predictions per reference
            identifier=identifier,
            skip_disconnected=skip_disconnected,  # skip disconnected ground truth graphs
            **job_kwargs,
        )
        df, metrics_cov = print_covmat_results(results, threshold=threshold)

        # Log results
        for k, v in metrics_cov.items():
            log.info(f"{k}: {v}")

        df.to_csv(os.path.join(save_folder, "covmat_results.csv"), index=False)
        np.save(os.path.join(save_folder, "covmat_rmsd_matrix.npy"), rmsd_matrix)
        with open(os.path.join(save_folder, "covmat_metrics.json"), "w") as f:
            json.dump({"Ratio": ratio, **metrics_cov}, f, indent=4)
    log.info("Inference completed.")


def run_covmat_evaluation(
    path_generated: str,
    path_dataset: Optional[str] = None,
    num_parallel: int = 8,
    worker_fn_type: str = "rmsd_rdkit_wo_h",
    threshold: float = 0.5,
    ratio: float = 2.0,
    save_folder: str = "covmat_evaluation_results",
    identifier: str = "smiles",
    skip_disconnected: bool = True,
    **job_kwargs,
):
    if not os.path.exists(path_generated):
        log.error(f"Generated conformers file not found: {path_generated}")
        return

    if ".pt" in path_generated:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Loading generated conformers from PyTorch file: {path_generated}")
        log.info(f"Using '{device}' for RMSD computation.")
        data_generated = torch.load(
            path_generated, weights_only=False, map_location=device
        )

        no_ref_conformers = sum(d.num_conformers.item() for d in data_generated)
        no_samples = sum(d.num_samples.item() for d in data_generated)

        log.info(
            f"Loaded {no_samples} generated conformers and {no_ref_conformers} "
            "reference conformers."
        )

        results, rmsd_matrix = evaluate_covmat(
            data_generated,
            thresholds=np.arange(0.05, 3.05, 0.05),
            num_parallel=num_parallel,
            worker_fn_type=worker_fn_type,
            ratio=ratio, # only keep at most ratio*n_conformers predictions per reference
            identifier=identifier,
            skip_disconnected=skip_disconnected, # skip disconnected ground truth graphs
            **job_kwargs,
        )

    elif ".xyz" in path_generated:
        if not os.path.exists(path_dataset):
            log.error(f"Dataset conformers file not found: {path_dataset}")
            return

        from ase.io import read

        log.info("Reading generated and dataset conformers from .xyz files...")

        log.info(f"Generated conformers path: {path_generated}")
        atoms_generated = read(path_generated, ":")

        log.info(f"Dataset conformers path: {path_dataset}")
        atoms_dataset = read(path_dataset, ":")

        log.info(
            f"Loaded {len(atoms_generated)} generated conformers and {len(atoms_dataset)} "
            "reference conformers."
        )

        kwargs_str = ", ".join(f"{k}={v}" for k, v in job_kwargs.items())
        log.info(
            f"Analysing coverage and matching (threshold: {threshold:.2f}, "
            f"ratio: {ratio:.0f}, num_parallel: {num_parallel}, "
            f"worker_fn_type: {worker_fn_type}, kwargs: {kwargs_str}):"
        )
        results, rmsd_matrix = evaluate_covmat(
            atoms_generated,
            atoms_dataset,
            thresholds=np.arange(0.05, 3.05, 0.05),
            num_parallel=num_parallel,
            worker_fn_type=worker_fn_type,
            ratio=ratio,  # only keep at most ratio*n_conformers predictions per reference
            identifier=identifier,
            skip_disconnected=skip_disconnected,  # skip disconnected ground truth graphs
            **job_kwargs,
        )
    df, metrics_cov = print_covmat_results(results, threshold=threshold)

    # Log results
    for k, v in metrics_cov.items():
        log.info(f"{k}: {v}")

    # Save results
    log.info(f"Saving results to folder: {save_folder}")
    os.makedirs(save_folder, exist_ok=True)
    df.to_csv(os.path.join(save_folder, "covmat_results.csv"), index=False)
    np.save(os.path.join(save_folder, "covmat_rmsd_matrix.npy"), rmsd_matrix)
    with open(os.path.join(save_folder, "covmat_metrics.json"), "w") as f:
        json.dump({"Ratio": ratio, **metrics_cov}, f, indent=4)

    log.info("Analysis completed.")
