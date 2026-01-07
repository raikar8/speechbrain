# Using Ground-Truth SNR and Noise Type Labels

If your training dataset includes ground-truth SNR and noise type labels, you can use them directly instead of estimating them. This provides more accurate routing and can improve model performance.

## JSON Dataset Format

Your JSON annotation files should include `snr_db` and `noise_type_id` fields:

```json
[
  {
    "id": "sample_001",
    "noisy_wav": "path/to/noisy.wav",
    "clean_wav": "path/to/clean.wav",
    "snr_db": 10.5,
    "noise_type_id": 1
  },
  ...
]
```

### Field Descriptions

- **snr_db**: Signal-to-Noise Ratio in decibels (float)
  - Example: `10.5`, `-2.3`, `15.0`
  
- **noise_type_id**: Noise type category ID (integer or string)
  - If integer: Direct ID (0 to num_noise_types-1)
  - If string: Will be mapped to ID using default mapping:
    - `"white"` → 0
    - `"babble"` → 1
    - `"street"` → 2
    - `"car"` → 3
    - `"other"` → 4

## Configuration

Enable label usage in your `hparams.yaml`:

```yaml
use_snr_noise_labels: True  # Enable use of ground-truth labels

modules:
  score_model: !new:speechbrain.integrations.models.sgmse_moe_noise.ScoreModelMoENoise
    K: 4
    num_experts: 25
    num_snr_bins: 5
    num_noise_types: 5
    # ... other parameters
```

## Benefits of Using Labels

1. **Accurate Routing**: Ground-truth labels provide exact SNR and noise type, eliminating estimation errors
2. **Better Training**: Model learns to route based on true noise characteristics
3. **Improved Performance**: More accurate expert selection leads to better enhancement quality
4. **Noise Classifier Training**: If `use_noise_classifier=True`, the classifier can be trained with supervision

## Custom Noise Type Mapping

If your dataset uses different noise type names, you can customize the mapping by modifying the `noise_type_map` in `train.py`:

```python
noise_type_map = {
    "white": 0,
    "babble": 1,
    "street": 2,
    "car": 3,
    "airport": 4,  # Custom type
    "station": 5,  # Custom type
    # ... add your mappings
}
```

**Note**: Make sure `num_noise_types` in your model config matches the number of unique noise types in your mapping.

## Fallback Behavior

If labels are not available in the JSON:
- The code will automatically fall back to estimation/classification
- Set `use_snr_noise_labels: False` to explicitly disable label usage
- The model will work with or without labels (backward compatible)

## Example JSON Entry

```json
{
  "id": "train_001",
  "noisy_wav": "{data_root}/train/noisy/train_001.wav",
  "clean_wav": "{data_root}/train/clean/train_001.wav",
  "snr_db": 12.3,
  "noise_type_id": "babble"
}
```

Or with integer noise type:

```json
{
  "id": "train_001",
  "noisy_wav": "{data_root}/train/noisy/train_001.wav",
  "clean_wav": "{data_root}/train/clean/train_001.wav",
  "snr_db": 12.3,
  "noise_type_id": 1
}
```

## Training with Labels

When labels are enabled:
1. Labels are extracted from JSON during data loading
2. Converted to tensors (SNR: float32, noise_type_id: int64)
3. Passed to model's forward method
4. Used directly for expert routing (no estimation needed)

The model automatically detects if labels are provided and uses them instead of estimation.

