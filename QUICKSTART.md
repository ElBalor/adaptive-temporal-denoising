# 🔬 Experiment 01: Adaptive Temporal Denoising Autoencoder

## Quick Start Guide

### Installation

```bash
cd experiments/01_adaptive_temporal_denoising
pip install -r requirements.txt
```

### Training

**Basic training (ECG dataset):**
```bash
python train.py --epochs 100 --batch_size 64
```

**Custom configuration:**
```bash
python train.py \
    --dataset audio \
    --epochs 150 \
    --batch_size 32 \
    --lr 5e-4 \
    --noise_type mixed \
    --snr_min 0 \
    --snr_max 20
```

**Resume from checkpoint:**
```bash
python train.py --resume results/checkpoints/latest.pt
```

### Visualization

After training, generate visualizations:
```bash
python visualize.py \
    --checkpoint results/checkpoints/best.pt \
    --dataset ecg \
    --save_dir results/figures
```

### TensorBoard Monitoring

During training, monitor in another terminal:
```bash
tensorboard --logdir results/logs
```

Then open: http://localhost:6006

---

## Architecture Overview

```
Noisy Input (B, 1, T)
        │
        ▼
┌───────────────────────────────────┐
│   MULTI-SCALE TCN ENCODER         │
│  ┌─────────┐ ┌─────────┐ ┌─────┐ │
│  │Scale 1  │ │Scale 2  │ │Scale│ │
│  │dilation1│ │dilation2│ │dilat│ │
│  └────┬────┘ └────┬────┘ └──┬──┘ │
│       └───────────┴──────────┘    │
│              │ CONCAT             │
└──────────────┼───────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │ NOISE ESTIMATOR     │──► noise_code (B, 16)
    └─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │ ATTENTION FUSION    │──► weights (B, 3)
    └─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │ TEMPORAL DECODER    │
    │ (conditioned)       │
    └─────────────────────┘
               │
               ▼
    Clean Output (B, 1, T)
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Scale Encoding** | 3 parallel TCN branches with different dilations |
| **Adaptive Noise Estimation** | Learns to predict noise level from latent features |
| **Attention Fusion** | Dynamically weights scale importance |
| **Spectral Loss** | Frequency-domain loss for better spectral fidelity |
| **Multi-Dataset Support** | ECG, Audio, Sensor data pipelines |

---

## Expected Results

After 100 epochs on ECG data:

| Metric | Target |
|--------|--------|
| Output SNR | >12 dB |
| Output SDR | >10 dB |
| SSIM | >0.85 |

---

## File Structure

```
01_adaptive_temporal_denoising/
├── model.py              # Neural network architecture
├── data.py               # Data pipelines & noise injection
├── train.py              # Training loop & logging
├── visualize.py          # Result visualization suite
├── requirements.txt      # Dependencies
└── results/
    ├── checkpoints/      # Saved models
    ├── figures/          # Generated plots
    └── logs/             # TensorBoard logs
```

---

## Configuration Options

### Datasets
- `ecg` - Synthetic ECG signals (default)
- `audio` - Speech-like signals with formants
- `sensor` - Human activity sensor data

### Noise Types
- `gaussian` - White Gaussian noise
- `poisson` - Shot noise
- `impulse` - Salt-pepper noise
- `colored` - 1/f^β noise
- `mixed` - Random combination (default, hardest)

### Model Hyperparameters
- `--hidden_channels` - Base channel width (default: 64)
- `--num_scales` - Number of parallel branches (default: 3)
- `--num_encoder_blocks` - TCN blocks per branch (default: 4)

---

## Tips for Better Results

1. **Start small** - Test with 10 epochs first to verify setup
2. **Monitor TensorBoard** - Watch for loss divergence
3. **Adjust SNR range** - Wider ranges = harder problem
4. **Use mixed noise** - Best for generalization
5. **Longer training** - 150+ epochs for best results

---

## Troubleshooting

**CUDA Out of Memory:**
```bash
python train.py --batch_size 32
```

**Loss not decreasing:**
- Lower learning rate: `--lr 1e-4`
- Check data: Run `python data.py` to test pipeline

**Slow training:**
- Enable cuDNN benchmarking
- Reduce sequence length: `--seq_len 512`

---

## Citation

If you use this code:

```bibtex
@software{temporal_signal_filter_2026,
  title = {Adaptive Temporal Denoising Autoencoder},
  author = {The Digital Necromancer},
  year = {2026},
  url = {https://github.com/yourusername/temporal-signal-filter}
}
```

---

<div align="center">

**"Noise is just signal waiting to be understood"**

*Part of the Temporal Signal Filter Lab*

</div>
