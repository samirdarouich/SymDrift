import logging
from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from symdrift.utils import sample_isotropic_Gaussian, sample_noise_like, RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

__all__ = [
    "CosineSchedule",
    "PolynomialSchedule",
    "NoiseSchedule",
    "GaussianDDPM",
    "VPGaussianDDPM",
    "KarrasEDMScheduler",
]


def clip_noise_schedule(
    alphas_bar: np.ndarray, clip_min: float = 0.0, clip_max: float = 1.0
) -> np.ndarray:
    """
    For a noise schedule given by alpha_bar, this clips alpha_t / alpha_t-1.
    This may help improve stability during sampling.

    Args:
        alphas_bar: noise schedule.
        clip_min: minimum value to clip to.
        clip_max: maximum value to clip to.
    """
    # get alphas from alphas_bar
    alphas = alphas_bar[1:] / alphas_bar[:-1]

    # clip alphas
    alphas = np.clip(alphas, a_min=clip_min, a_max=clip_max, dtype=np.float64)

    # recompute alphas_bar
    alphas_bar = np.cumprod(alphas, axis=0)

    return alphas_bar


def polynomial_decay(
    timesteps: int, s: float = 1e-5, clip_value: float = 0.001, power: float = 2.0
) -> np.ndarray:
    """
    A noise schedule based on a simple polynomial equation from
    https://arxiv.org/abs/2203.17003 to approximate the cosine schedule.

    Args:
        timesteps: number of timesteps T.
        s: precision parameter.
        clip_value: minimum value to clip to.
        power: power of the polynomial.
    """
    # compute alphas_bar
    t = np.linspace(0.0, 1.0, timesteps + 1, dtype=np.float64)
    alphas_bar = (1 - np.power(t, power)) ** 2

    # clip for more stable noise schedule
    alphas_bar = clip_noise_schedule(alphas_bar, clip_min=clip_value)

    # add precision
    precision = 1 - 2 * s
    alphas_bar = precision * alphas_bar + s

    return alphas_bar


def cosine_decay(
    timesteps, s: float = 0.008, v: float = 1.0, clip_value: float = 0.001
) -> np.ndarray:
    """
    Cosine schedule with clipping from https://arxiv.org/abs/2102.09672.

    Args:
        timesteps: number of timesteps T.
        s: precision parameter.
        v: decay parameter.
        clip_value: minimum value to clip to.
    """
    # compute alphas_bar
    t = np.linspace(0.0, 1.0, timesteps + 1, dtype=np.float64)
    f_t = np.cos((t**v + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_bar = f_t / f_t[0]

    # clip for more stable noise schedule
    alphas_bar = clip_noise_schedule(alphas_bar, clip_min=clip_value)

    return alphas_bar


def linear_decay(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> np.ndarray:
    """
    Linear schedule from https://arxiv.org/pdf/2006.11239.pdf.

    Args:
        timesteps: number of timesteps T.
        beta_start: starting value of beta_t, i.e. t=0.
        beta_end: ending value of beta_t, i.e. t=T.
    """
    # rescale beta_start and beta_end as the paper uses 1000 timesteps
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end

    # compute alphas_bar
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)

    return alphas_bar


class NoiseSchedule(nn.Module):
    """
    Base class for noise schedules. To be used together with Markovian processes,
    i.e. inheriting from ``MarkovianDiffusion``.
    """

    def __init__(
        self,
        T: int,
        alphas_bar: Union[np.ndarray, list],
        variance_type: str = "lower_bound",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            T: number of timesteps.
            alphas_bar: noise schedule.
            clip_value: minimum value to clip to.
            variance_type: use either 'lower_bound' or the 'upper_bound'.
            dtype: torch dtype to use for computation accuracy.
        """
        super().__init__()

        self.T = T
        self.alphas_bar = torch.tensor(alphas_bar, dtype=torch.float32, device="cpu")
        self.variance_type = variance_type
        self.dtype = dtype

        if isinstance(self.dtype, str):
            if self.dtype == "float64":
                self.dtype = torch.float64
            elif self.dtype == "float32":
                self.dtype = torch.float32
            else:
                raise ValueError(
                    f"data type must be float32 or float64, got {self.dtype}"
                )

        if self.variance_type == "upper_bound":
            logger.warning(
                "The upper bound for the posterior variance is not the exact one. "
                "This may affect the NLL estimation if used."
            )

        # pre-compute the parameters using double precision
        self.pre_compute_statistics()

    def pre_compute_statistics(self):
        """
        Pre-compute the noise parameters based on the notation of Ho et al.
        """

        self.betas_bar = 1 - self.alphas_bar
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_betas_bar = torch.sqrt(
            self.betas_bar
        )  # different from 1-sqrt(alphas_bar) !

        # infer the different statistics
        self.alphas = self.alphas_bar[1:] / self.alphas_bar[:-1]
        self.alphas = torch.concatenate([self.alphas_bar[:1], self.alphas])
        self.betas = 1.0 - self.alphas
        self.betas_square = self.betas**2
        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.inv_sqrt_alphas = 1.0 / self.sqrt_alphas
        self.inv_sqrt_betas_bar = 1.0 / self.sqrt_betas_bar

        # Weither to use the true posterior variance which is the lower bound
        # or the upper bound formulation
        if self.variance_type == "lower_bound":
            self.sigmas_square = self.betas[1:] * (
                self.betas_bar[:-1] / self.betas_bar[1:]
            )

            # lower bound sigma_1 = 0 because beta_bar_0 = 1 - alpha_bar_0 = 1 - 1 = 0
            # We clip to avoid inf when computing decoder loglikelihood or vlb weights
            self.sigmas_square = torch.concatenate(
                [self.sigmas_square[:1], self.sigmas_square]
            )

        elif self.variance_type == "upper_bound":
            self.sigmas_square = self.betas.clone()

            # we always replace the first value by the true posterior variacne
            # to have a better likelihood of L_0
            # see https://arxiv.org/abs/2102.09672
            self.sigmas_square[0] = self.betas[1] * (
                self.betas_bar[0] / self.betas_bar[1]
            )

        else:
            raise ValueError(
                "variance_type must be either 'lower_bound' or 'upper_bound'"
            )

        self.sigmas = torch.sqrt(self.sigmas_square)

        self.vlb_weights = self.betas**2 / (
            2 * self.sigmas_square * self.alphas * self.betas_bar
        )

    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the time t to [0, 1].

        Args:
            t: time steps.
        """
        if (t < 0).any() or (t >= self.T).any():
            raise ValueError(
                "t must be between 0 and T-1. This may be due to rounding errors. "
                "The indexing of the noise schedule starts with alpha_bar_1 at index 0."
            )

        return t.float() / (self.T - 1)

    def unnormalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Un-normalizes the time t to [0, T-1].

        Args:
            t: normalized time steps.
        """
        if (t < 0.0).any() or (t > 1.0).any():
            raise ValueError("t must be flaot between 0 and 1.")

        return torch.round(t.to(torch.double) * (self.T - 1)).long()

    def forward(self, t: torch.Tensor, keys: list = []) -> Dict[str, torch.Tensor]:
        """
        Query the noise parameters at timestep t.

        Args:
            t: the query timestep.
        """
        if not isinstance(t, torch.Tensor):
            raise ValueError("t must be a torch.Tensor.")

        device = t.device

        if len(t.shape) == 0:
            t = t.reshape(1)

        # convert to integer and numpy
        if t.dtype in [torch.float, torch.double]:
            t = self.unnormalize_time(t)

        t = t.to("cpu")

        # check if out of bounds
        if torch.any(t < 0) or torch.any(t >= self.T):
            raise ValueError(
                "t must be between 0 and T-1. This may be due to rounding errors. "
                "The indexing of the noise schedule starts with alpha_bar_1 at idnex 0."
            )

        # query the noise parameters
        key_to_attr_mapping = {
            "alpha_bar": self.alphas_bar[t],
            "beta_bar": self.betas_bar[t],
            "sqrt_alpha_bar": self.sqrt_alphas_bar[t],
            "sqrt_beta_bar": self.sqrt_betas_bar[t],
            "alpha": self.alphas[t],
            "beta": self.betas[t],
            "sqrt_alpha": self.sqrt_alphas[t],
            "sqrt_beta": self.sqrt_betas[t],
            "beta_square": self.betas_square[t],
            "sigma_square": self.sigmas_square[t],
            "sigma": self.sigmas[t],
            "inv_sqrt_alpha": self.inv_sqrt_alphas[t],
            "inv_sqrt_beta_bar": self.inv_sqrt_betas_bar[t],
            "vlb_weight": self.vlb_weights[t],
        }

        # return all keys if not specified
        if not keys:
            keys = key_to_attr_mapping.keys()  # type: ignore

        # fetch parameters and convert to torch tensors
        params = {}
        for key in keys:
            if key in key_to_attr_mapping:
                params[key] = key_to_attr_mapping[key].to(device=device).to(self.dtype)
            else:
                raise KeyError(
                    f"Key {key} not recognized. "
                    "If using continuous schedule, only few keys are available"
                    " because of the need of the next timestep for the others."
                )

        return params


class CosineSchedule(NoiseSchedule):
    """
    Cosine noise schedule.
    Subclasses ``NoiseSchedule``.
    """

    def __init__(
        self,
        T: int = 1000,
        s: float = 0.008,
        v: float = 1.0,
        clip_value: float = 0.001,
        variance_type: str = "lower_bound",
        **kwargs,
    ):
        """
        Args:
            T: number of steps.
            s: precision parameter.
            v: decay parameter.
            clip_value: clip parameetrs for numerical stability.
            variance_type: use either 'lower_bound' or the 'upper_bound'.
            kwargs: additional keyword arguments.
        """
        self.s = s
        self.v = v
        self.clip_value = clip_value
        alphas_bar = cosine_decay(T, s=self.s, v=self.v, clip_value=self.clip_value)

        super().__init__(
            T,
            alphas_bar,
            variance_type=variance_type,
            **kwargs,
        )


class PolynomialSchedule(NoiseSchedule):
    """
    Polynomial noise schedule.
    Subclasses ``NoiseSchedule``.
    """

    def __init__(
        self,
        T: int = 1000,
        s: float = 1e-5,
        clip_value: float = 0.001,
        variance_type: str = "lower_bound",
        **kwargs,
    ):
        """
        Args:
            T: number of steps.
            s: precision parameter.
            clip_value: clip parameetrs for numerical stability.
            variance_type: use either 'lower_bound' or the 'upper_bound'.
            kwargs: additional keyword arguments.
        """
        self.s = s
        self.clip_value = clip_value
        alphas_bar = polynomial_decay(T, s=s, clip_value=clip_value)

        super().__init__(
            T,
            alphas_bar,
            variance_type=variance_type,
            **kwargs,
        )


class LinearSchedule(NoiseSchedule):
    """
    Linear noise schedule.
    Subclasses ``NoiseSchedule``.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        variance_type: str = "lower_bound",
        **kwargs,
    ):
        """
        Args:
            T: number of steps.
            beta_start: starting value of beta_t, i.e. t=0.
            beta_end: ending value of beta_t, i.e. t=T.
            variance_type: use either 'lower_bound' or the 'upper_bound'.
            kwargs: additional keyword arguments.
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        alphas_bar = linear_decay(T, beta_start=beta_start, beta_end=beta_end)

        super().__init__(
            T,
            alphas_bar,
            variance_type=variance_type,
            **kwargs,
        )


class GaussianDDPM:
    """
    Base class for DDPM models using Gaussian diffusion kernels.
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
    ):
        """
        Args:
            noise_schedule: noise schedule for the diffusion process.
        """
        self.noise_schedule = noise_schedule

    @abstractmethod
    def perturbation_kernel(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        get the mean and std of the Gaussian perturbation kernel
        p(x_t|x_0) = N(mean(x_0,t),std(t)).

        Args:
            x_0: input tensor x_0 ~ p_data(x_0) to be diffused.
            t: time step.
        """
        raise NotImplementedError

    @abstractmethod
    def transition_kernel(
        self, x_t: torch.Tensor, t_next: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        get the mean and std of the transition kernel of the Gaussian Markov process,
        i.e. p(x_t+1|x_t) = N(mean(x_t,t+1),std(t+1)).

        Args:
            x_t: input tensor x_t ~ p_t at step t.
            t_next: next time step t+1.
            kwargs: additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def prior(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the Gaussian prior distribution p(x_T) = N(mean,std).

        Args:
            x: dummy input tensor, e.g. to infer shape.
            **kargs: additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def reverse_kernel(
        self,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the reverse transition kernel p(x_t-1|x_t)=N(mean(x_t,t),std(t))
        of the Markov process.

        Args:
            x_t: input tensor x_t ~ p_t at step t.
            noise: the (predicted) Gaussian noise to be removed.
            t: current time steps.
        """
        raise NotImplementedError

    def get_T(self) -> int:
        """
        Returns the total number of diffusion steps T.
        """
        return self.noise_schedule.T

    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the time t to [0, 1].

        Args:
            t: time steps as integer in [0, T-1].
        """
        return self.noise_schedule.normalize_time(t)

    def unnormalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Un-normalizes the time t to [0, T-1].

        Args:
            t: normalized time steps as float in [0, 1].
        """
        return self.noise_schedule.unnormalize_time(t)

    def forward_step(
        self,
        x_t: torch.Tensor,
        batch: Optional[torch.Tensor],
        t_next: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one Markov transition step to sample x_t+1 ~ p(x_t+1|x_t).

        Args:
            x_t: input tensor x_t ~ p_t at diffusion step t.
            batch: same as ``proporties.batch`` to assign each row of x to its system.
                    Set to None if one system or no invariance needed.
            t_next: next time steps t+1.
            kwargs: additional keyword arguments.
        """
        # get the mean and std of the transition kernel.
        mean, std = self.transition_kernel(x_t, t_next, **kwargs)

        # sample x_t+1.
        x_next, noise = sample_isotropic_Gaussian(mean, std, batch=batch, **kwargs)

        return x_next, noise

    def diffuse(
        self,
        x_0: torch.Tensor,
        batch: Optional[torch.Tensor],
        t: torch.Tensor,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Diffuses origin x_0 by t steps to sample x_t from p(x_t|x_0),
        given x_0 ~ p_data. Return tuple of tensors x_t and noise,
        or Dict of tensors with x_t and other quantities of interest.

        Args:
            x_0: input tensor x_0 ~ p_data to be diffused.
            batch: same as ``proporties.batch`` to assign each row to its system.
                Set to None if one system or no invariance needed.
            t: time steps.
            sample_key: key to store the diffused x_t.
            outpt_key: key to store the corresponding noise.
            return_dict: if True, return results under a dictionary of tensors.
            kwargs: additional keyword arguments.
        """
        # query noise parameters.
        mean, std = self.perturbation_kernel(x_0, t)

        # sample by Gaussian diffusion.
        x_t, noise = sample_isotropic_Gaussian(mean, std, batch=batch, **kwargs)

        return x_t, noise

    def sample_t_and_diffuse(
        self,
        x_0: torch.Tensor,
        batch: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples random time steps t and diffuses origin x_0 by t steps
        to sample x_t from p(x_t|x_0), given x_0 ~ p_data.
        Return tuple of tensors (x_t, t, noise).

        Args:
            x_0: input tensor x_0 ~ p_data to be diffused.
            batch: same as ``proporties.batch`` to assign each row to its system.
                Set to None if one system or no invariance needed.
            kwargs: additional keyword arguments.
        """

        # sample random time steps t
        batch_size = batch.max().item() + 1
        t = torch.randint(
            0,
            self.get_T(),
            size=(batch_size, 1),
            dtype=torch.long,
            device=x_0.device,
        )[batch]

        # diffuse x_0 to x_t
        x_t, noise = self.diffuse(x_0, batch, t, **kwargs)

        # normalize t to [0, 1]
        t = self.normalize_time(t)

        return x_t, t, noise

    def sample_prior(
        self, x: torch.Tensor, batch: Optional[torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Samples from the prior distribution p(x_T) = N(mean,std).

        Args:
            x: dummy input tensor, e.g. to infer shape.
            batch: same as ``proporties.batch`` to assign each row to its system.
            **kargs: additional keyword arguments.
        """
        # get the mean and std of the prior.
        mean, std = self.prior(x, **kwargs)

        # sample from the prior.
        x_T, _ = sample_isotropic_Gaussian(mean, std, batch=batch, **kwargs)

        return x_T

    def reverse_step(
        self,
        x_t: torch.Tensor,
        model_out: torch.Tensor,
        batch: Optional[torch.Tensor],
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Performs one reverse diffusion step to sample x_t-i ~ p(x_t-i|x_t), 0<i.

        Args:
            x_t: input tensor x_t ~ p_t at diffusion step t.
            model_out: output of the denoiser model.
            batch: same as ``proporties.batch`` to assign each row of x to its system.
                Set to None if one system or no invariance needed.
            t: time steps.
            **kwargs: additional keyword arguments for subclasses.
        """
        # get the noise prediction from the model output.
        noise = model_out

        # get the mean and std of the reverse transition kernel.
        mean, std = self.reverse_kernel(x_t, noise, t)

        # sample by Gaussian diffusion.
        x_t, _ = sample_isotropic_Gaussian(mean, std, batch=batch, **kwargs)

        return x_t


class VPGaussianDDPM(GaussianDDPM):
    """
    Variance Preserving DDPM model using Gaussian diffusion kernels.
    As proposed in HO et al. 2020 (https://arxiv.org/abs/2006.11239).
    """

    def __init__(self, noise_schedule: NoiseSchedule):
        """
        Args:
            noise_schedule: noise schedule to use for diffusion.
            kwargs: additional keyword arguments.
        """
        super().__init__(noise_schedule)

    def transition_kernel(
        self, x_t: torch.Tensor, t_next: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the transition kernel
        of the VP Gaussian Markov process,

        Args:
            x_t: input tensor x_t ~ p_t at step t.
            t_next: next time steps t+1.
            kwargs: additional keyword arguments.
        """
        # query noise parameters.
        noise_params = self.noise_schedule(t_next, keys=["sqrt_alpha", "sqrt_beta"])
        sqrt_alpha = noise_params["sqrt_alpha"]
        sqrt_beta = noise_params["sqrt_beta"]

        # get the mean and std of the transition kernel.
        mean = x_t * sqrt_alpha
        std = sqrt_beta

        return mean, std

    def perturbation_kernel(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the Gaussian perturbation kernel
        p(x_t|x_0) = N(mean(x_0,t),std(t)).

        Args:
            x_0: input tensor x_0 ~ p_data(x_0) to be diffused.
            t: time steps.
        """
        # query noise parameters.
        noise_params = self.noise_schedule(t, keys=["sqrt_alpha_bar", "sqrt_beta_bar"])
        sqrt_alpha_bar = noise_params["sqrt_alpha_bar"]
        sqrt_beta_bar = noise_params["sqrt_beta_bar"]

        # get the mean and std of the Gaussian perturbation kernel.
        mean = x_0 * sqrt_alpha_bar
        std = sqrt_beta_bar

        return mean, std

    def prior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the prior distribution p(x_T) = N(0,I).

        Args:
            x: dummy input tensor, e.g. to infer shape.
        """
        mean = torch.zeros_like(x)
        std = torch.ones_like(x)

        return mean, std

    def reverse_kernel(
        self,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the reverse transition kernel p(x_t-1|x_t)=N(mean(x_t-1,t),std(t))
        of the Markov process.

        Args:
            x_t: input tensor x_t ~ p_t at step t.
            noise: Gaussian noise to be reversed.
            t: current time steps.
        """
        # convert to correct dtype

        # get noise schedule parameters
        noise_params = self.noise_schedule(
            t,
            keys=["inv_sqrt_alpha", "beta", "inv_sqrt_beta_bar", "sigma"],
        )
        inv_sqrt_alpha_t = noise_params["inv_sqrt_alpha"]
        beta_t = noise_params["beta"]
        inv_sqrt_beta_t_bar = noise_params["inv_sqrt_beta_bar"]
        sigma_t = noise_params["sigma"]

        # add noise with variance \sigma (stochasticity) only for t!=0
        # otherwise return the mean \mu as the final sample
        sigma_t *= t != 0

        # compute the mean \mu of the reverse kernel
        mu = inv_sqrt_alpha_t * (x_t - (beta_t * inv_sqrt_beta_t_bar) * noise)

        return mu, sigma_t

    @torch.no_grad()
    def sample(
        self,
        num_steps,
        model,
        batch,
        conditioned=True,
        guidance_scale=0.0,
        t_start=None,
        x_start=None,
    ):

        if x_start is not None:
            assert t_start is not None, "t_start must be provided if x_start is given."
            x = x_start.clone()
        else:
            x = self.sample_prior(batch.pos, batch=batch.batch)
            t_start = self.get_T() - 1

        batch_size = x.size(0)
        trajectories = [x.clone()]
        timesteps = torch.linspace(
            t_start, 0, num_steps, dtype=torch.long, device=x.device
        )

        for i in timesteps:
            t = torch.full(
                (batch_size, 1), i, device=x.device
            )  # current timestep in integer
            batch.pos = x
            batch.t = self.normalize_time(t)  # normalized timestep in [0, 1]
            eps_pred = self.get_epsilon(
                model, batch, conditioned=conditioned, guidance_scale=guidance_scale
            )

            x = self.reverse_step(x, eps_pred, batch.batch, t)
            trajectories.append(x.clone())

        return x, trajectories

    @torch.no_grad()
    def get_epsilon(self, model, batch, conditioned=True, guidance_scale=0.0):
        if guidance_scale > 0.0:
            assert conditioned, "Classifier-free guidance requires conditional model."
            # Get both conditional and unconditional predictions
            eps_unconditioned = model(batch, conditioned=False)
            eps_conditioned = model(batch, conditioned=True)

            # Combine them using classifier-free guidance
            eps = eps_unconditioned + guidance_scale * (
                eps_conditioned - eps_unconditioned
            )
        else:
            eps = model(batch, conditioned=conditioned)

        return eps


class KarrasEDMScheduler:
    """
    Karras EDM scheduler operating in continuous sigma-space.

    This scheduler implements:
    - noising: x_t = x_0 + sigma * eps
    - sampling: Euler + 2nd-order correction in sigma-space
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
    ):
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be > 0, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max must be larger than sigma_min, got {sigma_max} <= {sigma_min}"
            )
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std

    def sample_sigma(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample per-graph sigma values from the EDM log-normal training distribution."""
        rnd = torch.randn(batch_size, device=device, dtype=dtype)
        return torch.exp(self.P_mean + self.P_std * rnd)

    def _get_sigma_schedule(
        self,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: Optional[float] = None,
    ) -> torch.Tensor:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        sigma_min = self.sigma_min if sigma_min is None else sigma_min
        sigma_max = self.sigma_max if sigma_max is None else sigma_max
        rho = self.rho if rho is None else rho

        if num_steps == 1:
            return torch.tensor([sigma_max, 0.0], device=device, dtype=dtype)

        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        sigma_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

        return torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])

    def perturbation_kernel(
        self, x_0: torch.Tensor, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean/std for p(x_t|x_0)=N(x_0, sigma^2 I)."""
        return x_0, sigma

    def diffuse(
        self,
        x_0: torch.Tensor,
        batch: Optional[torch.Tensor],
        sigma: Union[float, torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffuse x_0 at a given sigma to obtain x_t and the sampled noise."""
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=x_0.device, dtype=x_0.dtype)
        else:
            sigma = sigma.to(device=x_0.device, dtype=x_0.dtype)

        if sigma.ndim == 0:
            sigma_nodes = sigma.view(1, 1).expand(x_0.shape[0], 1)
        elif sigma.ndim == 1:
            if batch is None:
                if sigma.numel() != x_0.shape[0]:
                    raise ValueError(
                        "With batch=None, 1D sigma must have one entry per node."
                    )
                sigma_nodes = sigma[:, None]
            else:
                sigma_nodes = sigma[batch][:, None]
        elif sigma.ndim == 2:
            sigma_nodes = sigma
        else:
            raise ValueError(f"Unsupported sigma shape: {sigma.shape}")

        mean, std = self.perturbation_kernel(x_0, sigma_nodes)
        x_t, noise = sample_isotropic_Gaussian(mean, std, batch=batch, **kwargs)
        return x_t, noise

    def _predict_backbone(self, model, batch, conditioned=True, guidance_scale=0.0):
        """Predict model output with optional classifier-free guidance."""
        if guidance_scale > 0.0:
            eps_unconditioned = model(batch, conditioned=False)
            eps_conditioned = model(batch, conditioned=True)
            return eps_unconditioned + guidance_scale * (
                eps_conditioned - eps_unconditioned
            )

        try:
            return model(batch, conditioned=conditioned)
        except TypeError:
            return model(batch)

    def _denoise(self, model, batch, x: torch.Tensor, sigma_nodes: torch.Tensor):
        """Apply EDM preconditioning around a raw backbone model call."""
        sigma_sq = sigma_nodes**2
        sigma_data_sq = self.sigma_data**2
        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigma_nodes * self.sigma_data / torch.sqrt(sigma_sq + sigma_data_sq)
        c_in = 1.0 / torch.sqrt(sigma_sq + sigma_data_sq)
        c_noise = torch.log(sigma_nodes) / 4.0

        batch.pos = c_in * x
        batch.t = c_noise

        F_x = self._predict_backbone(model, batch)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x.to(x.dtype)

    @torch.no_grad()
    def sample(
        self,
        num_steps: int,
        model,
        batch,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: Optional[float] = None,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        x_start: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ):
        """Sample from EDM with Karras schedule using Euler + Heun correction."""
        if seed is not None:
            torch.manual_seed(seed)

        sigma_steps = self._get_sigma_schedule(
            num_steps=num_steps,
            device=batch.pos.device,
            dtype=dtype,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
        )

        if x_start is None:
            noise = sample_noise_like(batch.pos, batch.batch)
            x_next = noise.to(dtype) * sigma_steps[0]
        else:
            x_next = x_start.to(device=batch.pos.device, dtype=dtype)

        trajectories = [x_next.clone()] if return_trajectory else None
        for i, (sigma_cur, sigma_next) in enumerate(
            zip(sigma_steps[:-1], sigma_steps[1:])
        ):
            x_cur = x_next

            if S_churn > 0 and S_min <= sigma_cur <= S_max:
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
                sigma_hat = sigma_cur + gamma * sigma_cur
                x_hat = x_cur + torch.sqrt(
                    sigma_hat**2 - sigma_cur**2
                ) * S_noise * sample_noise_like(x_cur, batch.batch)
            else:
                sigma_hat = sigma_cur
                x_hat = x_cur

            sigma_hat_nodes = sigma_hat.view(1, 1).expand(x_hat.shape[0], 1)
            denoised_hat = self._denoise(model, batch, x_hat, sigma_hat_nodes)

            d_cur = (x_hat - denoised_hat) / torch.clamp(sigma_hat, min=1e-12)
            x_next = x_hat + (sigma_next - sigma_hat) * d_cur

            if i < num_steps - 1:
                sigma_next_nodes = sigma_next.view(1, 1).expand(x_next.shape[0], 1)
                denoised_next = self._denoise(
                    model,
                    batch,
                    x_next,
                    sigma_next_nodes,
                )
                d_prime = (x_next - denoised_next) / torch.clamp(
                    sigma_next, min=1e-12
                )
                x_next = x_hat + (sigma_next - sigma_hat) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

            if return_trajectory:
                trajectories.append(x_next.clone())

        if return_trajectory:
            return x_next, trajectories
        return x_next
