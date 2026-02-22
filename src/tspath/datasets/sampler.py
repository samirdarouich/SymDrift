from matplotlib.pylab import indices
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
        resample=False,
        seed=None,
    ):
        """
        Initialize a custom sampler for chunked dataset sampling.
        This sampler divides a dataset into fixed-size chunks based on composition groups,
        enabling efficient batch sampling with optional shuffling and resampling capabilities.
        Args:
            dataset: The dataset object containing samples organized by composition.
                     Must have attributes 'compositions' and 'comp_to_indices' (dict mapping
                     compositions to their sample indices).
            k (int): The number of chunks to sample in each iteration/batch.
            n (int): The size of each chunk (number of samples per chunk).
            shuffle (bool, optional): Whether to shuffle the chunk order during sampling.
                     Defaults to False.
            drop_last (bool, optional): Whether to drop the last incomplete batch if the
                     total number of chunks is not divisible by k. Defaults to False.
            resample (bool, optional): Whether to allow resampling of chunks (sampling with
                     replacement) across iterations. Defaults to False.
            seed (int, optional): Random seed for reproducibility. If None, no random seed
                     is set and sampling will be non-deterministic. Defaults to None.
        """
        self.dataset = dataset
        self.k = k
        self.n = n
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.resample = resample

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

        # --- Shuffle chunks ---
        if self.shuffle:
            perm = torch.randperm(self.total_chunks,generator=self.generator)
            chunk_order = perm.tolist()
        else:
            chunk_order = list(range(self.total_chunks))

        # --- Shuffle composition indices ---
        if self.shuffle:
            shuffled_indices = {}
            for comp in self.compositions:
                indices = self.dataset.comp_to_indices[comp]
                perm = torch.randperm(len(indices),generator=self.generator)
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
                chunk_items = shuffled_indices[comp][s:e]
                
                num_to_resample = self.n - len(chunk_items)
                if self.resample and num_to_resample > 0:
                    # In case too few samples in the chunk, sample randomly molecules 
                    # again until we have n samples.
                    extra_idx = torch.randint(
                        low=0, high=len(chunk_items), size=(num_to_resample,), 
                        generator=self.generator
                    )
                    extra_samples = [chunk_items[i] for i in extra_idx]
                    sampled = chunk_items + extra_samples
                else:
                    # take all items in the chunk (length ≤ n)
                    sampled = chunk_items
                batch_indices.extend(sampled)

            yield batch_indices