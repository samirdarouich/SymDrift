from functools import partial

import torch

from symdrift.alignment import (
    minimal_distance,
    minimal_distance_permuted,
    naive_distance,
)


class DriftingField:
    def __init__(
        self,
        temperatures=None,
        mask_self=True,
        normalize_over_x=False,
        normalize_drift=True,
        **kwargs,
    ):
        if temperatures is None:
            temperatures = torch.tensor([1.0])
        self.set_temperatures(temperatures)

        self.mask_self = mask_self
        self.normalize_over_x = normalize_over_x
        self.normalize_drift = normalize_drift

    def set_temperatures(self, temperatures):
        if isinstance(temperatures, (float, int)):
            temperatures = torch.tensor([temperatures])
        if isinstance(temperatures, list):
            temperatures = torch.tensor(temperatures)
        self.temperatures = temperatures

    def __call__(self, x, y_pos, y_neg, temperatures=None, **kwargs):
        """
        x: [N, D]
        y_pos: [N_pos, D]
        y_neg: [N_neg, D]
        temperatures: (T,) array of temperatures to use for each drift field (optional, if provided overrides self.temperatures)
        """
        if temperatures is not None:
            self.set_temperatures(temperatures)
        self.temperatures = self.temperatures.to(x.device)

        N, D = x.shape
        N_pos = y_pos.shape[0]
        N_neg = y_neg.shape[0]
        device = x.device

        # 1. Compute pairwise L2 distances
        diff_pos = y_pos[None, :, :] - x[:, None, :]  # (N, N_pos, D)
        diff_neg = y_neg[None, :, :] - x[:, None, :]  # (N, N_neg, D)
        dist_pos = torch.sqrt((diff_pos**2).sum(dim=2))  # (N, N_pos)
        dist_neg = torch.sqrt((diff_neg**2).sum(dim=2))  # (N, N_neg)

        # 2. Mask self-distances (when y_neg contains x)
        if self.mask_self and N == N_neg:
            mask = torch.eye(N, device=device) * 1e6
            dist_neg = dist_neg + mask

        # 3. Compute logits (normalize the temperature with sqrt of dimension D to keep
        # them transfearable between different dimensions.)
        logit_pos = -dist_pos.unsqueeze(0) / (
            self.temperatures[:, None, None] * D**0.5
        )  # (Ts, N, N_pos)
        logit_neg = -dist_neg.unsqueeze(0) / (
            self.temperatures[:, None, None] * D**0.5
        )  # (Ts, N, N_neg)

        # Compute kernel (normalize over y and optionally over x)
        w_pos = torch.softmax(logit_pos, dim=-1)  # softmax over y (columns)
        w_neg = torch.softmax(logit_neg, dim=-1)  # softmax over y (columns)
        if self.normalize_over_x:
            w_pos_ = torch.softmax(logit_pos, dim=-2)  # softmax over x (rows)
            w_neg_ = torch.softmax(logit_neg, dim=-2)  # softmax over x (rows)
            w_pos = torch.sqrt(w_pos * w_pos_)  # geometric mean
            w_neg = torch.sqrt(w_neg * w_neg_)  # geometric mean
            w_pos = w_pos / w_pos.sum(dim=-1, keepdim=True).clamp_min(
                1e-12
            )  # renormalize over y
            w_neg = w_neg / w_neg.sum(dim=-1, keepdim=True).clamp_min(
                1e-12
            )  # renormalize over y

        # 7. Compute drift as weighted average of differences (T is dim=0, x is dim=1, y is dim=2).
        # Aim is compute the drift for each molecule in x as a weighted average of the
        # differences to all molecules in y.
        drift_pos = (w_pos[..., None] * diff_pos).sum(dim=2)
        drift_neg = (w_neg[..., None] * diff_neg).sum(dim=2)
        V = drift_pos - drift_neg

        # 8. Combine V over different temperatures (T, N, D) to get final V (N, D)
        # The norm here includes the dimension D and the batch size N. This means that
        # v_norm = V.norm(dim=0) / sqrt(N*D)
        if self.normalize_drift:
            # normalize each temperature's V to have same norm
            v_norm = torch.sqrt(torch.mean(V**2, dim=(1, 2)))  # (T)
            v_pos_norm = torch.sqrt(torch.mean(drift_pos**2, dim=(1, 2)))  # (T)
            v_neg_norm = torch.sqrt(torch.mean(drift_neg**2, dim=(1, 2)))  # (T)

            V = V / (v_norm[:, None, None] + 1e-8)
            drift_pos = drift_pos / (v_pos_norm[:, None, None] + 1e-8)
            drift_neg = drift_neg / (v_neg_norm[:, None, None] + 1e-8)

        # sum over temperatures to get final V of shape (N, D)
        V = V.sum(dim=0)
        drift_pos = drift_pos.sum(dim=0)
        drift_neg = drift_neg.sum(dim=0)

        return V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg


class EquivariantDriftingField:
    def __init__(
        self,
        temperatures=None,
        mask_self=True,
        normalize_over_x=False,
        normalize_drift=True,
        aligned=True,
        permuted=True,
        brute_force_permutations=False,
        max_iter=3,
        tol=1e-2,
        rotate_before=True,
    ):
        if temperatures is None:
            temperatures = torch.tensor([1.0])
        self.set_temperatures(temperatures)
        self.mask_self = mask_self
        self.normalize_over_x = normalize_over_x
        self.normalize_drift = normalize_drift
        self.aligned = aligned
        self.permuted = permuted
        self.brute_force_permutations = brute_force_permutations
        self.max_iter = max_iter
        self.tol = tol
        self.rotate_before = rotate_before
        # Initialize distance function based on alignment and permutation settings
        self.get_distance_fn()

    def set_temperatures(self, temperatures):
        if isinstance(temperatures, (float, int)):
            temperatures = torch.tensor([temperatures])
        if isinstance(temperatures, list):
            temperatures = torch.tensor(temperatures)
        self.temperatures = temperatures

    def get_distance_fn(self):
        if self.aligned and self.permuted:
            self.distance_fn = partial(
                minimal_distance_permuted,
                brute_force_permutations=self.brute_force_permutations,
                max_iter=self.max_iter,
                tol=self.tol,
                rotate_before=self.rotate_before,
            )
        elif self.aligned and not self.permuted:
            self.distance_fn = minimal_distance
        elif not self.aligned and not self.permuted:
            self.distance_fn = naive_distance
        else:
            raise ValueError(
                "Permuted but not aligned doesn't make sense since permutation is only meaningful with alignment. Please set permuted=False if aligned=False."
            )

    def __call__(
        self,
        x,
        y_pos,
        y_neg,
        n_atoms,
        atomic_numbers_pos=None,
        atomic_numbers_neg=None,
        permutations_pos=None,
        permutations_neg=None,
        temperatures=None,
        aligned=None,
        permuted=None,
        brute_force_permutations=None,
    ):
        """
        x: (N*n_atoms, d)
        y_pos: (M*n_atoms, d)
        y_neg: (M*n_atoms, d)
        n_atoms: int
        atomic_numbers_pos: (M*n_atoms,) atomic numbers of each atom in y_pos,
            used to only permute within same atomic number (M*n_atoms,)
        atomic_numbers_neg: (M*n_atoms,) atomic numbers of each atom in y_neg,
            used to only permute within same atomic number (M*n_atoms,)
        permutations_pos: (P, n_atoms) precomputed permutations to apply to y_pos.
        permutations_neg: (P, n_atoms) precomputed permutations to apply to y_neg.
        temperatures: (T,) array of temperatures to use for each drift field (optional, if provided overrides self.temperatures)
        aligned: bool (optional, if provided overrides self.aligned and updates distance function)
        permuted: bool (optional, if provided overrides self.permuted and updates distance function)
        brute_force_permutations: bool (optional, if provided overrides self.brute_force_permutations and updates distance function)
        returns: (N*n_atoms, d) drifting field for each molecule in x
        """
        _, d = x.shape
        assert y_pos.shape[1] == d, (
            f"Expected y_pos to have {d} dimensions, got {y_pos.shape[1]}"
        )

        if temperatures is not None:
            self.set_temperatures(temperatures)
        self.temperatures = self.temperatures.to(x.device)

        if aligned is not None:
            self.aligned = aligned
            if permuted is not None:
                self.permuted = permuted
            if brute_force_permutations is not None:
                self.brute_force_permutations = brute_force_permutations
            self.get_distance_fn()

        # 0. Reshape to (N, n_atoms, d) and (N_pos/N_neg, n_atoms, d)
        x_ = x.view(-1, n_atoms, d)  # (N, n_atoms, d)
        y_pos_ = y_pos.view(-1, n_atoms, d)  # (M, n_atoms, d)
        y_neg_ = y_neg.view(-1, n_atoms, d)  # (M, n_atoms, d)

        N = x_.shape[0]
        N_pos = y_pos_.shape[0]
        N_neg = y_neg_.shape[0]
        device = x.device

        # 1. Compute alignment-aware pairwise L2 distances and returns (N, N_pos/N_neg) RMSD
        # matrix and aligned difference y-x of shape (N, N_pos/N_neg, n_atoms, d)
        if atomic_numbers_pos is not None:
            assert atomic_numbers_pos.shape == (N_pos * n_atoms,), (
                f"Expected atomic_numbers_pos to have shape {(N_pos * n_atoms,)}, got {atomic_numbers_pos.shape}"
            )
            atomic_numbers_pos = atomic_numbers_pos.view(N_pos, n_atoms)
            if atomic_numbers_neg is None:
                if N_pos != N_neg:
                    # assume negative samples are just repeated positive samples
                    # (e.g. for each positive sample we have x negative sample which is
                    # the same molecule but with different noise)
                    atomic_numbers_neg = atomic_numbers_pos.repeat_interleave(
                        N_neg // N_pos, dim=0
                    )
            else:
                assert atomic_numbers_neg.shape == (N_neg * n_atoms,), (
                    f"Expected atomic_numbers_neg to have shape {(N_neg * n_atoms,)}, got {atomic_numbers_neg.shape}"
                )
                atomic_numbers_neg = atomic_numbers_neg.view(N_neg, n_atoms)

        if permutations_pos is not None:
            assert permutations_pos.shape[1] == n_atoms, (
                f"Expected permutations_pos to have shape (P, {n_atoms}), got {permutations_pos.shape}"
            )
            if permutations_neg is None:
                permutations_neg = permutations_pos

        # Distances are RMSD, hence they are normalized by sqrt of number of atoms, so
        # that they are transfearable between different molecule sizes.
        dist_pos, diff_pos = self.distance_fn(
            x_, y_pos_, atomic_numbers=atomic_numbers_pos, permutations=permutations_pos
        )
        dist_neg, diff_neg = self.distance_fn(
            x_, y_neg_, atomic_numbers=atomic_numbers_neg, permutations=permutations_neg
        )

        # 2. Mask self-distances (when y_neg contains x)
        if self.mask_self and N == N_neg:
            mask = torch.eye(N, device=device) * 1e6
            dist_neg = dist_neg + mask

        # 3. Compute logits
        logit_pos = (
            -dist_pos.unsqueeze(0) / self.temperatures[:, None, None]
        )  # (Ts, N, N_pos)
        logit_neg = (
            -dist_neg.unsqueeze(0) / self.temperatures[:, None, None]
        )  # (Ts, N, N_neg)

        # Compute kernel (normalize over y and optionally over x) (T, N, N_pos/N_neg)
        w_pos = torch.softmax(logit_pos, dim=-1)  # softmax over y (columns)
        w_neg = torch.softmax(logit_neg, dim=-1)  # softmax over y (columns)
        if self.normalize_over_x:
            w_pos_ = torch.softmax(logit_pos, dim=-2)  # softmax over x (rows)
            w_neg_ = torch.softmax(logit_neg, dim=-2)  # softmax over x (rows)
            w_pos = torch.sqrt(w_pos * w_pos_)  # geometric mean
            w_neg = torch.sqrt(w_neg * w_neg_)  # geometric mean
            w_pos = w_pos / w_pos.sum(dim=-1, keepdim=True).clamp_min(
                1e-12
            )  # renormalize over y
            w_neg = w_neg / w_neg.sum(dim=-1, keepdim=True).clamp_min(
                1e-12
            )  # renormalize over y

        # 7. Compute drift as weighted average of differences (T is dim=0, x is dim=1, y is dim=2).
        # Aim is compute the drift for each molecule in x as a weighted average of the
        # differences to all molecules in y. --> (T, N, n_atoms, d)
        drift_pos = (w_pos[..., None, None] * diff_pos.unsqueeze(0)).sum(dim=2)
        drift_neg = (w_neg[..., None, None] * diff_neg.unsqueeze(0)).sum(dim=2)
        V = drift_pos - drift_neg

        # 8. Combine V over different temperatures (T, N, n_atoms, d) to get final V (N, n_atoms, d)
        # The norm here includes the dimension D (n_atoms*d) and the batch size N. This means that
        # v_norm = V.norm(dim=0) / sqrt(N*D)
        if self.normalize_drift:
            # normalize each temperature's V to have same norm
            v_norm = torch.sqrt(torch.mean(V**2, dim=(1, 2, 3)))  # (T)
            v_pos_norm = torch.sqrt(torch.mean(drift_pos**2, dim=(1, 2, 3)))  # (T)
            v_neg_norm = torch.sqrt(torch.mean(drift_neg**2, dim=(1, 2, 3)))  # (T)

            V = V / (v_norm[:, None, None, None] + 1e-8)
            drift_pos = drift_pos / (v_pos_norm[:, None, None, None] + 1e-8)
            drift_neg = drift_neg / (v_neg_norm[:, None, None, None] + 1e-8)

        # sum over temperatures to get final V of shape (N, n_atoms, d)
        V = V.sum(dim=0)
        drift_pos = drift_pos.sum(dim=0)
        drift_neg = drift_neg.sum(dim=0)

        # Norming each V before combining equally weights each temperature's contribution
        # to the final drift. This also means, that the overall drift direction is different
        # if the norms between the different drifs is very different. This is a design
        # choice.

        # 9. reshape V to (N*n_atoms, d)
        V = V.view_as(x)
        drift_pos = drift_pos.view_as(x)
        drift_neg = drift_neg.view_as(x)

        return V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg
