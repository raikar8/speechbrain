# SGMSE with Mixture of Experts

This implementation adds Mixture of Experts (MoE) capability to SGMSE with two routing strategies:
1. **Version 1**: Timestep-based routing (Remix-DiT style)
2. **Version 2**: Noise-based routing (environmental noise characteristics)

## Overview

### Version 1: Timestep-Based Routing (Remix-DiT Style)

This version implements timestep-based expert routing, where different experts are activated based on the diffusion timestep `t`.

**Key Features:**
- **Basis Models**: Trains only `K` basis models (e.g., K=4) instead of `N` independent experts
- **Learnable Mixing**: Creates `N` experts (e.g., N=20) by mixing `K` basis models using learnable coefficients
- **Timestep Routing**: Routes to different experts based on diffusion timestep `t`
- **Efficient**: Maintains same inference efficiency as single model (one expert active per timestep)

## Architecture

### Basis Model Approach

Instead of training `N` independent expert models, we:
1. Train `K` basis models (where `K < N`)
2. Create `N` experts by mixing `K` basis models with learnable coefficients
3. Route based on timestep `t` to select the appropriate expert

### Mixing Mechanism

For each timestep `t`:
1. Discretize `t` into bins (e.g., 20 bins)
2. Map bin to expert ID
3. Get mixing coefficients for that expert: `mix_weights = mixing_coefficients[expert_id]` (shape: `K`)
4. Mix basis model outputs: `output = sum(mix_weights[k] * basis_models[k](x_t, y, t) for k in range(K))`

### Learnable Coefficients

The mixing coefficients are learnable parameters:
- Shape: `(num_experts, K)` matrix
- Each row represents mixing weights for one expert
- Initialized with small random values
- Updated during training via backpropagation

## Usage

### Configuration

Use the provided `hparams_moe.yaml` or modify your existing `hparams.yaml`:

```yaml
modules:
  score_model: !new:speechbrain.integrations.models.sgmse_moe.ScoreModelMoE
    backbone: ncsnpp_v2
    sde: ouve
    K: 4                    # Number of basis models
    num_experts: 20         # Number of experts to create
    num_timestep_bins: 20   # Number of timestep bins
    # ... other parameters same as ScoreModel
```

### Training

Training works exactly the same as the original SGMSE:

```bash
python train.py --hparams hparams_moe.yaml
```

The training loop automatically handles:
- EMA updates for all basis models
- Mixing coefficient updates
- Timestep-based routing

### Key Parameters

- **K**: Number of basis models to train (default: 4)
  - Lower K = less training cost, but less model diversity
  - Recommended: 4-8 for good balance

- **num_experts**: Number of experts to create via mixing (default: 20)
  - Should be >= K
  - Typically matches number of timestep bins
  - More experts = more specialization, but more parameters

- **num_timestep_bins**: Number of bins to discretize timestep `t` (default: num_experts)
  - Determines granularity of timestep-based routing
  - More bins = finer-grained routing

## Implementation Details

### Timestep Discretization

Timesteps are discretized into bins:
```python
t_normalized = (t - t_eps) / (T - t_eps)  # Normalize to [0, 1]
bin_index = (t_normalized * (num_timestep_bins - 1)).long()
expert_id = bin_index % num_experts
```

### Forward Pass

1. Route to expert based on timestep `t`
2. Get mixing weights for that expert
3. Forward pass through all `K` basis models
4. Mix outputs using learned coefficients
5. Apply scaling and loss-specific transformations (same as ScoreModel)

### EMA Handling

EMA is updated for:
- All parameters of all `K` basis models
- Mixing coefficients

This ensures stable training and evaluation.

## Efficiency Considerations

### Training
- **Parameters**: `K * backbone_params + (num_experts * K)` mixing coefficients
- **Forward pass**: All `K` basis models are evaluated (necessary for mixing)
- **Memory**: Slightly higher than single model due to `K` models

### Inference
- **Parameters**: Same as training
- **Forward pass**: All `K` basis models are evaluated (mixing requires all outputs)
- **Efficiency**: Similar to single model since mixing is just weighted sum

**Note**: Unlike Remix-DiT which can precompute mixed models, we mix on-the-fly to handle variable timesteps during sampling.

## Comparison with Original SGMSE

| Aspect | Original SGMSE | SGMSE-MoE (Version 1) |
|--------|----------------|----------------------|
| Models | 1 backbone | K basis models |
| Routing | None | Timestep-based |
| Parameters | Backbone params | K × backbone + mixing coeffs |
| Training Cost | Baseline | ~K × baseline |
| Inference Cost | Baseline | ~K × baseline |
| Specialization | Single model for all timesteps | Different experts for different timesteps |

## Expected Benefits

Based on Remix-DiT findings:
1. **Better Quality**: Experts specialize for different timesteps (early = coarse structure, late = fine details)
2. **Adaptive Capacity**: Learnable coefficients allocate more capacity where needed
3. **Efficient Training**: Train K models instead of N independent models

### Version 2: Noise-Based Routing

This version implements noise-based expert routing, where different experts are activated based on environmental noise characteristics (SNR, noise type).

**Key Features:**
- **Basis Models**: Trains only `K` basis models (e.g., K=4) instead of `N` independent experts
- **Learnable Mixing**: Creates `N` experts (e.g., N=25) by mixing `K` basis models using learnable coefficients
- **Noise-Based Routing**: Routes to different experts based on:
  - **SNR**: Signal-to-Noise Ratio (discretized into bins, e.g., -5 to 20 dB)
  - **Noise Type**: Type of environmental noise (white, babble, street, car, etc.)
- **Noise Classification**: Optional learned classifier for noise type, or heuristic-based

**Architecture:**

1. **Noise Estimation**:
   - SNR: Estimated from noisy vs clean signal (if available) or heuristically
   - Noise Type: Classified using learned MLP or spectral characteristics

2. **Expert Routing**:
   - Map (SNR_bin, noise_type) → expert_id
   - Expert grid: `expert_id = noise_type_id * num_snr_bins + snr_bin`

3. **Mixing**:
   - Same basis model mixing as Version 1
   - Mixing coefficients learned per (SNR, noise_type) combination

**Usage:**

```yaml
modules:
  score_model: !new:speechbrain.integrations.models.sgmse_moe_noise.ScoreModelMoENoise
    K: 4                    # Number of basis models
    num_experts: 25         # Number of experts (should be >= num_snr_bins * num_noise_types)
    num_snr_bins: 5         # Number of SNR bins
    num_noise_types: 5       # Number of noise type categories
    use_noise_classifier: True  # Use learned classifier
    snr_range: [-5, 20]     # SNR range in dB
    # ... other parameters same as ScoreModel
```

**Key Parameters:**
- **num_snr_bins**: Number of SNR bins (default: 5)
  - More bins = finer-grained SNR routing
- **num_noise_types**: Number of noise type categories (default: 5)
  - Categories: white, babble, street, car, etc.
- **use_noise_classifier**: Whether to use learned classifier (default: True)
  - True: Train MLP to classify noise type
  - False: Use heuristic based on spectral characteristics
- **snr_range**: SNR range in dB for binning (default: [-5, 20])

**Expected Benefits:**
- Experts specialize for different noise conditions
- Better handling of diverse environmental noise
- Improved generalization across noise types and SNR levels

## Future Versions

- **Version 3**: Hybrid routing (combine timestep and noise-based routing)

## References

1. Fang, G., Ma, X., & Wang, X. (2024). Remix-DiT: Mixing Diffusion Transformers for Multi-Expert Denoising. arXiv preprint arXiv:2412.05628.

2. Richter, J., Welker, S., Lemercier, J.-M., Lay, B., & Gerkmann, T. (2023). Speech Enhancement and Dereverberation with Diffusion-based Generative Models. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31, 2351-2364.

