# MoE Versions Comparison

This document compares the three MoE implementations for SGMSE.

## Overview

| Version | Routing Strategy | File | Key Feature |
|---------|-----------------|------|-------------|
| **Version 1** | Timestep-based | `sgmse_moe.py` | Routes by diffusion timestep `t` |
| **Version 2** | Noise-based | `sgmse_moe_noise.py` | Routes by environmental noise (SNR, noise type) |
| **Version 3** | Hybrid | `sgmse_moe_hybrid.py` | Routes by BOTH timestep AND noise characteristics |

## Detailed Comparison

### Version 1: Timestep-Based Routing (Remix-DiT Style)

**Routing Dimension:** Diffusion timestep `t` only

**Expert Mapping:**
- Discretize `t` into `num_timestep_bins` bins
- Map bin → expert_id
- `expert_id = t_bin % num_experts`

**Use Case:**
- When you want experts to specialize for different stages of the diffusion process
- Early timesteps (high noise) vs late timesteps (low noise) need different capabilities
- Follows Remix-DiT approach directly

**Configuration:**
```yaml
score_model: !new:speechbrain.integrations.models.sgmse_moe.ScoreModelMoE
  K: 4
  num_experts: 20
  num_timestep_bins: 20
```

**Expected Experts:** `num_experts` (typically 20)

---

### Version 2: Noise-Based Routing

**Routing Dimensions:** SNR + Noise Type

**Expert Mapping:**
- Discretize SNR into `num_snr_bins` bins
- Classify noise type into `num_noise_types` categories
- `expert_id = (snr_bin * num_noise_types) + noise_type_id`

**Use Case:**
- When you want experts to specialize for different environmental noise conditions
- Different noise types (white, babble, street) need different handling
- Different SNR levels require different strategies

**Configuration:**
```yaml
score_model: !new:speechbrain.integrations.models.sgmse_moe_noise.ScoreModelMoENoise
  K: 4
  num_experts: 25
  num_snr_bins: 5
  num_noise_types: 5
  use_noise_classifier: True
  snr_range: [-5, 20]
```

**Expected Experts:** `num_snr_bins * num_noise_types` (typically 25)

**Supports Labels:** Yes - can use ground-truth SNR and noise_type_id from dataset

---

### Version 3: Hybrid Routing

**Routing Dimensions:** Timestep + SNR + Noise Type

**Expert Mapping (Grid Strategy):**
- Discretize `t` into `num_timestep_bins` bins
- Discretize SNR into `num_snr_bins` bins
- Classify noise type into `num_noise_types` categories
- `expert_id = (t_bin * num_snr_bins * num_noise_types) + (snr_bin * num_noise_types) + noise_type_id`

**Expert Mapping (Hierarchical Strategy):**
- First route by noise: `noise_expert = snr_bin * num_noise_types + noise_type_id`
- Then route by timestep: `expert_id = noise_expert * num_timestep_bins + t_bin`

**Use Case:**
- Maximum specialization: Experts can specialize for specific (timestep, SNR, noise_type) combinations
- Best of both worlds: Combines benefits of both routing strategies
- Fine-grained control: Most granular expert selection

**Configuration:**
```yaml
score_model: !new:speechbrain.integrations.models.sgmse_moe_hybrid.ScoreModelMoEHybrid
  K: 4
  num_experts: 100
  num_timestep_bins: 10
  num_snr_bins: 5
  num_noise_types: 5
  routing_strategy: grid  # or "hierarchical"
  use_noise_classifier: True
  snr_range: [-5, 20]
```

**Expected Experts:** `num_timestep_bins * num_snr_bins * num_noise_types` (typically 250 for full coverage)

**Supports Labels:** Yes - can use ground-truth SNR and noise_type_id from dataset

---

## Parameter Comparison

| Parameter | Version 1 | Version 2 | Version 3 |
|-----------|-----------|-----------|-----------|
| `K` (basis models) | ✓ | ✓ | ✓ |
| `num_experts` | ✓ | ✓ | ✓ |
| `num_timestep_bins` | ✓ | ✗ | ✓ |
| `num_snr_bins` | ✗ | ✓ | ✓ |
| `num_noise_types` | ✗ | ✓ | ✓ |
| `use_noise_classifier` | ✗ | ✓ | ✓ |
| `snr_range` | ✗ | ✓ | ✓ |
| `routing_strategy` | ✗ | ✗ | ✓ |

## When to Use Which Version?

### Use Version 1 (Timestep-Based) if:
- You want to follow Remix-DiT approach directly
- You're primarily interested in timestep specialization
- You don't have noise labels or don't want to estimate noise characteristics
- Simpler setup is preferred

### Use Version 2 (Noise-Based) if:
- You have diverse noise conditions in your dataset
- You want experts to specialize for different noise types/SNR levels
- You have (or can estimate) noise characteristics
- You want to handle environmental noise variations

### Use Version 3 (Hybrid) if:
- You want maximum specialization and control
- You have both timestep and noise information available
- You want the best possible performance (at cost of more experts)
- You're willing to train with more experts

## Computational Cost

| Version | Training Cost | Inference Cost | Parameters |
|---------|--------------|----------------|------------|
| Version 1 | ~K × baseline | ~K × baseline | K × backbone + (N × K) mixing |
| Version 2 | ~K × baseline | ~K × baseline | K × backbone + (N × K) mixing + classifier |
| Version 3 | ~K × baseline | ~K × baseline | K × backbone + (N × K) mixing + classifier |

**Note:** All versions evaluate all K basis models during forward pass (necessary for mixing). The cost is similar across versions, but Version 3 may need more experts (N) for full coverage.

## Example Expert Counts

For a typical setup:
- **Version 1**: 20 experts (10-20 timestep bins)
- **Version 2**: 25 experts (5 SNR bins × 5 noise types)
- **Version 3**: 250 experts (10 timestep × 5 SNR × 5 noise types) for full coverage

You can use fewer experts in Version 3 (e.g., 100) and let combinations share experts via modulo.

## All Versions Support

- ✅ Basis model approach (train K, create N experts)
- ✅ Learnable mixing coefficients
- ✅ EMA for all basis models
- ✅ Same interface as ScoreModel
- ✅ Compatible with existing training code
- ✅ Ground-truth labels (Version 2 & 3)

