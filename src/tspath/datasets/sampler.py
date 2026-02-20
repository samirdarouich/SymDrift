import torch
import math
from torch.utils.data import Sampler

class CompositionBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        k,
        n,
        shuffle=False,
        drop_last=False,
        seed=None,
    ):
        self.dataset = dataset
        self.k = k
        self.n = n
        self.shuffle = shuffle
        self.drop_last = drop_last

        if seed is None:
            self.generator = None
        else:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

        self.compositions = list(dataset.compositions)

        # ---- Precompute chunk metadata ----
        # Store for each chunk:
        # (composition, start_idx, end_idx)
        self.chunk_specs = []

        for comp in self.compositions:
            indices = dataset.comp_to_indices[comp]
            m = len(indices)
            num_chunks = math.ceil(m / n)

            for i in range(num_chunks):
                start = i * n
                end = min((i + 1) * n, m)
                self.chunk_specs.append((comp, start, end))

        self.total_chunks = len(self.chunk_specs)

    def __len__(self):
        if self.drop_last:
            return self.total_chunks // self.k
        return math.ceil(self.total_chunks / self.k)

    def __iter__(self):

        # Optionally shuffle chunk order
        if self.shuffle:
            perm = torch.randperm(
                self.total_chunks,
                generator=self.generator
            )
            chunk_order = perm.tolist()
        else:
            chunk_order = list(range(self.total_chunks))

        # Optional per-composition molecule shuffle
        if self.shuffle:
            shuffled_indices = {}
            for comp in self.compositions:
                indices = self.dataset.comp_to_indices[comp]
                perm = torch.randperm(
                    len(indices),
                    generator=self.generator
                )
                shuffled_indices[comp] = [indices[i] for i in perm]
        else:
            shuffled_indices = self.dataset.comp_to_indices

        num_batches = len(self)

        for b in range(num_batches):
            start = b * self.k
            end = start + self.k
            selected = chunk_order[start:end]

            if len(selected) < self.k and self.drop_last:
                break

            batch_indices = []

            for chunk_id in selected:
                comp, s, e = self.chunk_specs[chunk_id]
                batch_indices.extend(shuffled_indices[comp][s:e])

            yield batch_indices