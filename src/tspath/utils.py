import itertools
import logging
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import rich
import torch
import yaml
from ase import Atoms
from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree
from torch_scatter import scatter_mean

__all__ = [
    "print_config",
    "batch_center_systems",
    "get_canonical_elementwise_permutations",
    "get_brute_force_permutations",
    "get_x_y_pairs",
    "inputs_to_atoms",
    "batch_inputs_to_atoms",
    "sample_noise",
    "sample_noise_like",
    "sample_noise_like_2d",
    "sample_isotropic_Gaussian",
    "RankedLogger",
    "Queue",
]


def empty(*args, **kwargs):
    pass


def todict(config: Union[DictConfig, Dict]):
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    return config_dict


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "run",
        "globals",
        "data",
        "model",
        "task",
        "trainer",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree(
        ":gear: Running with the following config:", style=style, guide_style=style
    )

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)


def batch_center_systems(systems: torch.Tensor, batch: torch.Tensor, dim: int = 0):
    """
    center batch of systems moleculewise to have zero center of geometry

    Args:
        systems (torch.tensor): batch of systems (molecules)
        batch (torch.tensor): the system id for each atom in the batch
        dim (int): dimension to scatter over
    """

    # Compute mean position per molecule
    mean = scatter_mean(systems, batch, dim=dim)

    # broadcast mean to the same shape as systems
    mean = mean.movedim(dim, 0)[batch].movedim(0, dim)

    return systems - mean


def get_canonical_elementwise_permutations(canonical_atomic_numbers):
    """
    canonical_atomic_numbers: (n_atoms,) sorted

    Returns
    -------
    perms : (P, n_atoms)
    """
    device = canonical_atomic_numbers.device

    unique_elements = []
    for a in canonical_atomic_numbers.tolist():
        if a not in unique_elements:
            unique_elements.append(a)

    element_indices = [
        torch.where(canonical_atomic_numbers == elem)[0].tolist()
        for elem in unique_elements
    ]

    element_perms = [
        list(itertools.permutations(indices)) for indices in element_indices
    ]

    all_perms = []
    for prod in itertools.product(*element_perms):
        perm = list(itertools.chain(*prod))
        all_perms.append(perm)

    return torch.tensor(all_perms, device=device)  # (P,n)


def get_brute_force_permutations(x, y, atomic_numbers=None):
    """
    Canonicalized permutation pipeline
    """
    device = x.device
    B, n, d = x.shape

    # -------------------------------------------------
    # 1) Canonicalize atom ordering
    # -------------------------------------------------
    if atomic_numbers is not None:
        sort_idx = torch.argsort(atomic_numbers, dim=1)
        inv_sort_idx = torch.argsort(sort_idx, dim=1)

        gather_idx = sort_idx[..., None].expand(-1, -1, d)

        x = torch.gather(x, 1, gather_idx)
        y = torch.gather(y, 1, gather_idx)

        canonical_atomic_numbers = atomic_numbers[0].sort().values

        perms = get_canonical_elementwise_permutations(
            canonical_atomic_numbers
        )  # (P,n)

    else:
        perms = torch.tensor(list(itertools.permutations(range(n))), device=device)

        sort_idx = None
        inv_sort_idx = None

    P = perms.shape[0]

    # -------------------------------------------------
    # 2) Apply permutations
    # -------------------------------------------------

    perms_exp = perms[None, :, :, None].expand(B, P, n, d)

    y_exp = y[:, None].expand(B, P, n, d)

    y_perm = torch.gather(y_exp, 2, perms_exp)

    # -------------------------------------------------
    # 3) Flatten batch
    # -------------------------------------------------

    x_flat = x[:, None].expand(B, P, n, d).reshape(B * P, n, d)
    y_flat = y_perm.reshape(B * P, n, d)

    return x_flat, y_flat, perms, sort_idx, inv_sort_idx


def get_x_y_pairs(x, y, atomic_numbers=None):
    """
    x: (N, n_atoms, d),
    y: (M, n_atoms, d)
    atomic_numbers: (n_atoms,) atomic numbers of each atom in target structure. Only
    permute within same atomic number if provided.

    returns:
    x_flat, y_flat of shape (N*M, n_atoms, d) for all pairs
    """
    assert x.shape[1] == y.shape[1], "X and Y must have same number of atoms"
    N, n_atoms, d = x.shape
    M = y.shape[0]

    # Create all N x M pairs and flatten to (N*M, n_atoms, d)
    x_pairs = x[:, None, :, :].expand(N, M, n_atoms, d)
    y_pairs = y[None, :, :, :].expand(N, M, n_atoms, d)
    x_flat = x_pairs.reshape(N * M, n_atoms, d)
    y_flat = y_pairs.reshape(N * M, n_atoms, d)

    atomic_numbers_b = None
    if atomic_numbers is not None:
        assert atomic_numbers.shape == (M * n_atoms,), (
            f"Atomic numbers should have shape (M*n_atoms,) not {atomic_numbers.shape}"
        )
        atomic_numbers_b = atomic_numbers.view(M, n_atoms).repeat(
            N, 1
        )  # (N*M, n_atoms)

    return x_flat, y_flat, atomic_numbers_b


def inputs_to_atoms(inputs, pos_key="pos", info_keys=[]):
    """
    Converts a single input to an ASE Atoms object.

    Args:
        inputs: The input in PyTorch geometric format.
        pos_key (str): The key in the input that contains the positions.
    Returns:
        Atoms: The ASE Atoms object.
    """
    R = inputs[pos_key].detach().cpu().numpy()
    Z = inputs.x.detach().cpu().numpy()
    info = {}
    for key in info_keys:
        if hasattr(inputs, key):
            item_ = inputs[key]
            if isinstance(item_, str):
                info[key] = item_
            else:
                info[key] = item_.item()
    atoms = Atoms(positions=R, numbers=Z, info=info)
    return atoms


def batch_inputs_to_atoms(batch, pos_key="pos", info_keys=[]):
    """
    Converts a batch of inputs to a list of ASE Atoms objects.

    Args:
        batch: The input batch in PyTorch geometric format.
        pos_key (str): The key in the batch that contains the positions.
    Returns:
        List[Atoms]: The list of ASE Atoms objects.
    """
    atoms_list = []

    for m in batch.batch.unique():
        mask = batch.batch == m
        R = batch[pos_key][mask].detach().cpu().numpy()
        Z = batch.x[mask].detach().cpu().numpy()
        info = {}
        for key in info_keys:
            if hasattr(batch, key):
                item_ = batch[key][m]
                if isinstance(item_, str):
                    info[key] = item_
                else:
                    info[key] = item_.item()
        atoms = Atoms(positions=R, numbers=Z, info=info)
        atoms_list.append(atoms)
    return atoms_list


def sample_noise(
    shape: Tuple,
    batch: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Sample Gaussian noise based on input shape.
    Project to the zero center of geometry if invariant.

    Args:
        shape: shape of the noise.
        batch: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    """
    # sample noise
    noise = torch.randn(shape, device=device, dtype=dtype)

    # The invariance trick: project noise to the zero center of geometry.
    # system-wise center of geometry
    if batch is not None:
        noise = batch_center_systems(noise, batch, dim=-2)  # type: ignore
    # global center of geometry if one system passed.
    else:
        noise -= noise.mean(-2).unsqueeze(-2)

    return noise


def sample_noise_like(
    x: torch.Tensor,
    batch: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Sample Gaussian noise based on input x.

    Args:
        x: input tensor, e.g. to infer shape.
        batch: same as ``batch.batch`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
    """
    return sample_noise(x.shape, batch, device=x.device, dtype=x.dtype)


def sample_noise_like_2d(pos: torch.Tensor, batch: torch.Tensor):
    """
    Sample 2d Gaussian noise and add zero z-component.
    Center the noise to have zero center of geometry.
    """
    z = torch.randn(pos.shape[0], 2, device=pos.device)
    z = batch_center_systems(z, batch)  # zero center of geometry
    z = torch.cat([z, torch.zeros(z.shape[0], 1, device=z.device)], dim=1)
    return z


def sample_isotropic_Gaussian(
    mean: torch.Tensor,
    std: torch.Tensor,
    batch: Optional[torch.Tensor],
    noise: Optional[torch.Tensor] = None,
):
    """
    Use the reparametrization trick to Sample from iso Gaussian distribution
    with given mean and std.

    Args:
        mean: mean of the Gaussian distribution.
        std: standard deviation of the Gaussian distribution.
        batch: same as ``proporties.batch`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
        noise: the Gaussian noise. If None, a new noise is sampled.
    """
    # sample noise if not given.
    if noise is None:
        noise = sample_noise_like(mean, batch)

    # sample using the Gaussian reparametrization trick.
    sample = mean + std * noise

    return sample, noise


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(
        self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs
    ) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError(
                    "The `rank_zero_only.rank` needs to be set before use"
                )
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)

# Gradient clipping
class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)