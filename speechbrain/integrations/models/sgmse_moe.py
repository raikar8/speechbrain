"""
Remix-DiT style Mixture of Experts for SGMSE.

This module implements timestep-based routing using basis models and learnable mixing coefficients,
following the approach from "Remix-DiT: Mixing Diffusion Transformers for Multi-Expert Denoising".

References:
[1] Fang, G., Ma, X., & Wang, X. (2024). Remix-DiT: Mixing Diffusion Transformers for Multi-Expert Denoising.
    arXiv preprint arXiv:2412.05628.
[2] Richter, J., Welker, S., Lemercier, J.-M., Lay, B., & Gerkmann, T. (2023).
    Speech Enhancement and Dereverberation with Diffusion-based Generative Models.
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31, 2351-2364.
"""

from math import ceil

import sgmse.sampling as sampling
import torch
import torch.nn as nn
from sgmse.backbones import BackboneRegistry
from sgmse.sdes import SDERegistry
from torch_ema import ExponentialMovingAverage
from torch_pesq import PesqLoss


class ScoreModelMoE(nn.Module):
    """
    Score-based generative model with Mixture of Experts (Remix-DiT style).
    
    Creates N experts from K basis models using learnable mixing coefficients.
    Routes to experts based on diffusion timestep t.
    
    Key features:
    - Trains only K basis models (K < N), reducing training cost
    - Creates N experts by mixing K basis models with learnable coefficients
    - Routes based on timestep t (different experts for different noise levels)
    - Maintains same inference efficiency as single model (one expert active per timestep)
    
    Arguments
    ---------
    backbone: str
        Name of the backbone network architecture.
    sde: str
        Identifier of the SDE to use for diffusion sampling.
    K: int
        Number of basis models to train (default: 4).
    num_experts: int
        Number of expert models to create via mixing (default: 20).
        Should be >= K. Typically num_experts = number of timestep bins.
    num_timestep_bins: int
        Number of bins to discretize timestep t into (default: 20).
        If None, uses N.
    lr: float
        Learning rate for optimizer.
    ema_decay: float
        Exponential moving average decay rate.
    t_eps: float
        Minimum time offset for numerical stability.
    num_eval_files: int
        Number of files to evaluate during validation.
    loss_type: str
        One of "score_matching", "denoiser", or "data_prediction".
    loss_weighting: str
        Weighting scheme for the loss (e.g., "sigma^2").
    network_scaling: str or None
        Scaling applied to network output.
    c_in: str
    c_out: str
    c_skip: str
        Coefficients for signal combinations.
    sigma_data: float
        Data noise standard deviation for EDM.
    l1_weight: float
        Weight for L1 term in data_prediction loss.
    pesq_weight: float
        Weight for PESQ loss term.
    sr: int
        Sample rate of audio.
    num_frames: int
        Number of time-frequency frames.
    hop_length: int
        Hop length between frames.
    **kwargs
        Arguments for creation of backbone.
    """

    def __init__(
        self,
        backbone="ncsnpp_v2",
        sde="ouve",
        K=4,  # Number of basis models
        num_experts=20,  # Number of experts to create
        num_timestep_bins=None,  # If None, uses num_experts
        lr=1e-4,
        ema_decay=0.999,
        t_eps=0.03,
        num_eval_files=20,
        loss_type="score_matching",
        loss_weighting="sigma^2",
        network_scaling=None,
        c_in="1",
        c_out="1",
        c_skip="0",
        sigma_data=0.1,
        l1_weight=0.001,
        pesq_weight=0.0,
        sr=16000,
        num_frames=256,
        hop_length=128,
        **kwargs,
    ):
        super().__init__()
        
        # MoE hyperparameters
        self.K = K  # Number of basis models
        self.num_experts = num_experts  # Number of experts
        self.num_timestep_bins = num_timestep_bins if num_timestep_bins is not None else num_experts
        
        if num_experts < K:
            raise ValueError(f"num_experts ({num_experts}) must be >= K ({K})")
        
        # Initialize Backbone DNN - create K basis models
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.basis_models = nn.ModuleList([
            dnn_cls(**kwargs) for _ in range(K)
        ])
        
        # Learnable mixing coefficients: (num_experts, K) matrix
        # Each row represents mixing weights for one expert
        # Initialized with small random values
        self.mixing_coefficients = nn.Parameter(
            torch.randn(num_experts, K) * 0.1
        )
        
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        
        # Save hyperparams
        self.lr = lr
        self.ema_decay = ema_decay
        # EMA for all basis models
        all_params = []
        for basis_model in self.basis_models:
            all_params.extend(list(basis_model.parameters()))
        all_params.append(self.mixing_coefficients)  # Also track mixing coefficients
        self.ema = ExponentialMovingAverage(
            all_params, decay=self.ema_decay
        )
        self._error_loading_ema = False

        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.num_frames = num_frames
        self.hop_length = hop_length
        self.sr = sr
        self.l1_weight = l1_weight
        self.pesq_weight = pesq_weight
        
        # Get T from SDE for timestep binning
        self.T = getattr(self.sde, 'T', 1.0)
        
        # PESQ loss, if used
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False

    def _get_timestep_bin(self, t):
        """
        Discretize timestep t into bins for routing.
        
        Arguments
        ---------
        t: torch.Tensor
            Timestep tensor of shape (B,).
            
        Returns
        -------
        bin_indices: torch.Tensor
            Bin indices of shape (B,) with values in [0, num_timestep_bins-1].
        """
        # Normalize t to [0, 1] range (assuming t is in [t_eps, T])
        t_normalized = (t - self.t_eps) / (self.T - self.t_eps)
        t_normalized = torch.clamp(t_normalized, 0.0, 1.0)
        
        # Convert to bin indices
        bin_indices = (t_normalized * (self.num_timestep_bins - 1)).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_timestep_bins - 1)
        
        return bin_indices

    def _get_expert_id(self, t):
        """
        Map timestep bin to expert ID.
        
        Arguments
        ---------
        t: torch.Tensor
            Timestep tensor of shape (B,).
            
        Returns
        -------
        expert_ids: torch.Tensor
            Expert IDs of shape (B,) with values in [0, num_experts-1].
        """
        bin_indices = self._get_timestep_bin(t)
        # Map bin to expert (can be 1-to-1 or many-to-one)
        # For simplicity, use modulo to map bins to experts
        expert_ids = bin_indices % self.num_experts
        return expert_ids

    def _mix_basis_models(self, expert_id, x_t, y, t):
        """
        Mix basis models to create expert output.
        
        Arguments
        ---------
        expert_id: torch.Tensor
            Expert IDs of shape (B,).
        x_t: torch.Tensor
            The perturbed spectrogram at time `t`, of shape (B, 1, F, T).
        y: torch.Tensor
            The noisy input spectrogram of shape (B, 1, F, T).
        t: torch.Tensor
            The time step, of shape (B,).
            
        Returns
        -------
        mixed_output: torch.Tensor
            Mixed output from basis models, shape (B, 1, F, T).
        """
        B = x_t.shape[0]
        device = x_t.device
        
        # Get mixing coefficients for each sample's expert
        # expert_id: (B,), mixing_coefficients: (num_experts, K)
        # We need to index mixing_coefficients with expert_id
        mix_weights = self.mixing_coefficients[expert_id]  # (B, K)
        
        # Get outputs from all basis models
        basis_outputs = []
        for basis_model in self.basis_models:
            if self.backbone == "ncsnpp_v2":
                output = basis_model(
                    self._c_in(t) * x_t,
                    self._c_in(t) * y,
                    t
                )
            else:
                dnn_input = torch.cat([x_t, y], dim=1)
                output = -basis_model(dnn_input, t)
            basis_outputs.append(output)
        
        # Stack basis outputs: (B, K, 1, F, T)
        basis_outputs = torch.stack(basis_outputs, dim=1)
        
        # Apply mixing: mix_weights (B, K), basis_outputs (B, K, 1, F, T)
        # We need to expand mix_weights to match dimensions
        mix_weights = mix_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1, 1)
        mixed_output = torch.sum(mix_weights * basis_outputs, dim=1)  # (B, 1, F, T)
        
        return mixed_output

    def forward(self, x_t, y, t):
        """
        Computes the score or predicted clean data using MoE routing.
        
        Arguments
        ---------
        x_t: torch.Tensor
            The perturbed spectrogram at time `t`, of shape (B, 1, F, T).
        y: torch.Tensor
            The noisy input spectrogram of shape (B, 1, F, T).
        t: torch.Tensor
            The time step, of shape (B,).
            
        Returns
        -------
        torch.Tensor
            The computed score or the predicted clean data `x_hat`,
            depending on `self.loss_type`. Shape is (B, 1, F, T).
        """
        # Route to expert based on timestep
        expert_id = self._get_expert_id(t)
        
        # Get mixed output from basis models
        F = self._mix_basis_models(expert_id, x_t, y, t)
        
        # Scaling the network output, see below Eq. (7) in the paper
        if self.network_scaling == "1/sigma":
            std = self.sde._std(t)
            F = F / std[:, None, None, None]
        elif self.network_scaling == "1/t":
            F = F / t[:, None, None, None]
        
        # The loss type determines the output of the model
        if self.loss_type == "score_matching":
            score = self._c_skip(t) * x_t + self._c_out(t) * F
            return score
        elif self.loss_type == "denoiser":
            sigmas = self.sde._std(t)[:, None, None, None]
            score = (F - x_t) / sigmas.pow(2)
            return score
        elif self.loss_type == "data_prediction":
            x_hat = self._c_skip(t) * x_t + self._c_out(t) * F
            return x_hat
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[
                :, None, None, None
            ]
        else:
            raise ValueError(f"Invalid c_in type: {self.c_in}")

    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None]
        elif self.c_out == "edm":
            sigma = self.sde._std(t)
            return (
                (sigma * self.sigma_data)
                / torch.sqrt(self.sigma_data**2 + sigma**2)
            )[:, None, None, None]
        else:
            raise ValueError(f"Invalid c_out type: {self.c_out}")

    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        elif self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[
                :, None, None, None
            ]
        else:
            raise ValueError(f"Invalid c_skip type: {self.c_skip}")

    def get_pc_sampler(
        self,
        predictor_name,
        corrector_name,
        y,
        N=None,
        minibatch=None,
        **kwargs,
    ):
        """Get a predictor-corrector sampler for the SGMSE MoE model."""
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name,
                corrector_name,
                sde=sde,
                score_fn=self,
                y=y,
                **kwargs,
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                """Batched sampling function for large inputs."""
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_pc_sampler(
                        predictor_name,
                        corrector_name,
                        sde=sde,
                        score_fn=self,
                        y=y_mini,
                        **kwargs,
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        """Get an ODE sampler for the SGMSE MoE model."""
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                """Batched sampling function for large inputs."""
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_ode_sampler(
                        sde, self, y=y_mini, **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns

            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        """Get a Schrödinger bridge sampler for the SGMSE MoE model."""
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(
            sde, self, y=y, sampler_type=sampler_type, **kwargs
        )

    def enhance(
        self,
        y,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        timeit=False,
        **kwargs,
    ):
        """
        One-call speech enhancement from a noisy input using MoE.
        
        This method runs the chosen SGMSE sampler to produce an enhanced spectrogram
        from the input `y`, using timestep-based expert routing.
        """
        # SGMSE sampling with OUVE SDE
        if self.sde.__class__.__name__ == "OUVESDE":
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(
                    predictor,
                    corrector,
                    y.cuda(),
                    N=N,
                    corrector_steps=corrector_steps,
                    snr=snr,
                    intermediate=False,
                    **kwargs,
                )
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(y.cuda(), N=N, **kwargs)
            else:
                raise ValueError(
                    f"Invalid sampler type for SGMSE sampling: {sampler_type}"
                )
        # Schrödinger bridge sampling with VE SDE
        elif self.sde.__class__.__name__ == "SBVESDE":
            sampler = self.get_sb_sampler(
                sde=self.sde, y=y.cuda(), sampler_type=self.sde.sampler_type
            )
        else:
            raise ValueError(
                f"Invalid SDE type for speech enhancement: {self.sde.__class__.__name__}"
            )
        sample, _ = sampler()
        return sample

    def compute_loss(
        self,
        forward_out,
        x_t,
        z,
        t,
        mean,
        x,
        reduction="mean",
        to_audio_func=None,
    ):
        """
        Compute the loss for the score-based generative model with MoE.
        
        Same as ScoreModel.compute_loss, but works with MoE outputs.
        """
        sigma = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            score = forward_out
            if self.loss_weighting == "sigma^2":
                losses = torch.square(torch.abs(score * sigma + z))  # Eq. (7)
            else:
                raise ValueError(
                    f"Invalid loss weighting for loss_type=score_matching: {self.loss_weighting}"
                )
            per_sample_loss = 0.5 * torch.sum(
                losses.reshape(losses.shape[0], -1), dim=-1
            )

        elif self.loss_type == "denoiser":
            score = forward_out
            D = score * sigma.pow(2) + x_t  # equivalent to Eq. (10)
            losses = torch.square(torch.abs(D - mean))  # Eq. (8)
            if self.loss_weighting == "1":
                pass
            elif self.loss_weighting == "sigma^2":
                losses = losses * sigma**2
            elif self.loss_weighting == "edm":
                losses = (
                    (sigma**2 + self.sigma_data**2)
                    / ((sigma * self.sigma_data) ** 2)
                )[:, None, None, None] * losses
            else:
                raise ValueError(
                    f"Invalid loss weighting for loss_type=denoiser: {self.loss_weighting}"
                )
            per_sample_loss = 0.5 * torch.sum(
                losses.reshape(losses.shape[0], -1), dim=-1
            )

        elif self.loss_type == "data_prediction":
            if to_audio_func is None:
                raise ValueError(
                    "to_audio_func must be provided for data prediction loss"
                )

            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = (1 / (F * T)) * torch.square(torch.abs(x_hat - x))
            losses_tf = 0.5 * torch.sum(
                losses_tf.reshape(losses_tf.shape[0], -1), dim=-1
            )

            # losses in the time domain (td)
            target_len = (self.num_frames - 1) * self.hop_length
            x_hat_td = to_audio_func(x_hat.squeeze(), target_len)
            x_td = to_audio_func(x.squeeze(), target_len)
            losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
            losses_l1 = 0.5 * torch.sum(
                losses_l1.reshape(losses_l1.shape[0], -1), dim=-1
            )

            if self.pesq_weight > 0.0:
                losses_pesq = self.pesq_loss(x_td, x_hat_td)
                losses_pesq = torch.mean(
                    losses_pesq
                )
                per_sample_loss = (
                    losses_tf
                    + self.l1_weight * losses_l1
                    + self.pesq_weight * losses_pesq
                )
            else:
                per_sample_loss = losses_tf + self.l1_weight * losses_l1
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        if reduction == "mean":
            return torch.mean(per_sample_loss)
        elif reduction == "none":
            return per_sample_loss
        else:
            raise ValueError("Invalid reduction type")

    def update_ema(self):
        """Call this after each optimizer step to update the EMA weights."""
        all_params = []
        for basis_model in self.basis_models:
            all_params.extend(list(basis_model.parameters()))
        all_params.append(self.mixing_coefficients)
        self.ema.update(all_params)

    def store_ema(self):
        """Call this before evaluation if you want to switch to EMA weights."""
        all_params = []
        for basis_model in self.basis_models:
            all_params.extend(list(basis_model.parameters()))
        all_params.append(self.mixing_coefficients)
        self.ema.store(all_params)
        self.ema.copy_to(all_params)

    def restore_ema(self):
        """Call this after evaluation if you stored EMA weights and want to restore normal weights."""
        all_params = []
        for basis_model in self.basis_models:
            all_params.extend(list(basis_model.parameters()))
        all_params.append(self.mixing_coefficients)
        self.ema.restore(all_params)

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

