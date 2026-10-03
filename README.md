# 🔬 Experiment 1: Adaptive Temporal Denoising Autoencoder

## Status: **READY FOR PRODUCTION**

| Component | Status | Notes |
|-----------|--------|-------|
| Model Architecture | ✅ Complete | Multi-scale TCN + attention |
| Loss Functions | ✅ Complete | Composite loss (MSE + Spectral) |
| Training Loop | ✅ Complete | Advanced monitoring |
| Evaluation Suite | ✅ Complete | Failure mode detection |
| Synthetic Data | ✅ Complete | ECG, Audio, Sensor |
| Real Data Loaders | ✅ Complete | PhysioNet, LibriSpeech, UCI HAR |
| Visualization | ✅ Complete | Signal, spectral, attention plots |
| Documentation | ✅ Complete | All guides written |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training (Synthetic Only — Fast Start)
```bash
python train.py --epochs 100 --batch_size 64 --dataset ecg
```

### 3. Training (With Real Data — Full Power)
```bash
# First download PhysioNet PTB-XL (see DATA_SETUP.md)
python train.py --epochs 100 --batch_size 64 --use_real --real_ratio 0.3 --real_data_dir ./data/physionet
```

### 4. Monitor Training
```bash
tensorboard --logdir results/logs
```

### 5. Evaluate
```bash
python evaluate.py --checkpoint results/checkpoints/best.pt
python visualize.py --checkpoint results/checkpoints/best.pt --dataset ecg
```

---

## 📊 Architecture

```
Noisy Input (B, 1, T)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  MULTI-SCALE TCN ENCODER (WeightNorm "Muscle" + RMSNorm "Nerves")│
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ Scale 1     │ │ Scale 2     │ │ Scale 3     │               │
│  │ (Fast)      │ │ (Medium)    │ │ (Deep)      │               │
│  │ dilation 1  │ │ dilation 2  │ │ dilation 4  │               │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘               │
│         └───────────────┴───────────────┘                       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  NOISE CONDITION ESTIMATOR ──► noise_code (B, 16)               │
│  Estimates noise level from encoded features                    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  POINT-WISE ATTENTION FUSION (Dilation-Gate)                    │
│  Attention per timestep, conditioned on noise level             │
│  Output: fused features (B, D, T) + attn_weights (B, 3, T)      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  🔥 TRINITY BOTTLENECK                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ ScaledDotProductAttention (8 heads, strictly causal)      │  │
│  │       ↓                                                    │  │
│  │ PositionwiseMLP (Conv1d → RMSNorm → GELU → Conv1d)        │  │
│  └───────────────────────────────────────────────────────────┘  │
│  Transformer-level processing in temporal domain                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  STRICTLY CAUSAL DECODER (WeightNorm + RMSNorm)                 │
│  Noise-conditioned injection → 4 Conv stages → Output           │
│  Real-time ready: y_t depends ONLY on x_{≤t}                    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
              Clean Output (B, 1, T)
```

**Parameters:** ~1.2-1.5M (upgraded from 0.53M)
**Key Upgrades:** RMSNorm, WeightNorm, Point-wise Attention, Trinity Bottleneck, Strict Causality

---

## 🎯 Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Multi-Scale TCN Encoding** | 3 parallel branches with progressive dilation | Captures features at Fast, Medium, Deep temporal scales |
| **WeightNorm ("Muscle")** | Weight normalization on all convolutions | Stabilized training, faster convergence |
| **RMSNorm ("Nerves")** | Root Mean Square layer normalization | Lightweight, handles signal spikes efficiently |
| **Point-Wise Attention Fusion** | Per-timestep attention over scales, noise-conditioned | Model focuses on signal-rich temporal points |
| **Dilation-Gate Logic** | Attention conditioned on estimated noise level | Adaptive fusion strength based on noise severity |
| **Trinity Bottleneck** | ScaledDotProductAttention (8 heads) + PositionwiseMLP | Transformer-level temporal processing |
| **Strict Causality** | Output at time t depends ONLY on input at times ≤t | Real-time ready, no future leakage |
| **Adaptive Noise Estimation** | Learns to predict noise level from latent features | No manual tuning needed |
| **Spectral Loss** | Complex L1 loss in frequency domain (magnitude + phase) | Preserves signal identity, not just MSE |
| **Real Data Support** | PhysioNet PTB-XL (21,837 records), LibriSpeech, UCI HAR | Works on real-world data, not just synthetic |

---

## 📈 Expected Results

### Synthetic Data Only
| Metric | Epoch 5 | Epoch 50 | Epoch 100 |
|--------|---------|----------|-----------|
| SNR | 4-6 dB | 8-10 dB | 10-12 dB |
| Spectral Sim | 0.85 | 0.88 | 0.90 |
| Smoothness | 0.6 | 0.75 | 0.80 |

### With Real Data (30% mix)
| Metric | Epoch 5 | Epoch 50 | Epoch 100 |
|--------|---------|----------|-----------|
| SNR | 6-8 dB | 12-14 dB | **14-16 dB** |
| Spectral Sim | 0.88 | 0.93 | **0.95** |
| Smoothness | 0.7 | 0.85 | **0.90** |

---

## 🔍 Advanced Usage

### Cross-Domain Transfer Test
```bash
# Train on ECG
python train.py --epochs 100 --dataset ecg --use_real --real_data_dir ./data/physionet

# Test on Audio (generalization check)
python evaluate.py --checkpoint results/checkpoints/best.pt --test-dataset audio
```

### Ablation Study
```bash
# No attention (concat only)
# Edit model.py: remove feature_fusion, use concat directly

# No noise conditioning
# Edit model.py: remove noise_estimator, use zeros for noise_code

# Single scale
python train.py --num_scales 1
```

### Hyperparameter Tuning
```bash
# Larger model
python train.py --hidden_channels 128 --num_encoder_blocks 6

# Faster training
python train.py --batch_size 128 --seq_len 512

# Harder noise
python train.py --snr_min 0 --snr_max 15
```

---

## 📁 File Structure

```
01_adaptive_temporal_denoising/
├── model.py              # Multi-scale TCN autoencoder
├── data.py               # Synthetic + real data loaders
├── train.py              # Training loop with monitoring
├── evaluate.py           # Comprehensive evaluation suite
├── visualize.py          # Result visualization
├── test_system.py        # System verification tests
├── requirements.txt      # Dependencies
├── README.md            # This file
├── DATA_SETUP.md        # Real data download guide
├── QUICKSTART.md        # Quick start guide
├── EVALUATION_GUIDE.md  # Evaluation philosophy
└── results/
    ├── checkpoints/      # Saved models
    ├── figures/          # Generated plots
    └── logs/             # TensorBoard logs
```

---

## 🧪 System Verification

Run before training:
```bash
python test_system.py
```

Expected output:
```
✅ Model test PASSED
✅ Loss function test PASSED
✅ Metrics test PASSED
✅ Data pipeline test PASSED

🚀 System is READY FOR TRAINING
```

---

## 📊 Monitoring During Training

**Console output every epoch:**
```
Epoch  12 | Train Loss: 1.1245 | Val Loss: 1.0892 | Val SNR: 7.42 dB | Val SDR: 7.38 dB
Spec Sim: 0.892 | Smooth: 0.85 | Low-SNR: 5.1 dB
Time: 41.2s | Total: 0.01h | Best Val Loss: 1.0892 | Best Val SNR: 7.42 dB
```

**Watch for warnings:**
- `⚠️ OVER-SMOOTHING` → Model is blurring (increase spectral weight)
- `⚠️ LOW spectral similarity` → Identity loss (model distorting signal)
- `⚠️ POOR low-SNR performance` → Not working at 0-5 dB (need more training or harder noise)

---

## 🎯 Paper Potential

**What makes this novel:**

1. **Transformer-Grade Temporal Processing** — Trinity Bottleneck (8-head self-attention + MLP) in a denoising autoencoder
2. **Point-Wise Attention Fusion** — Per-timestep attention over temporal scales, not global weights
3. **Strictly Causal Architecture** — Real-time ready with WeightNorm + RMSNorm stabilization
4. **Dilation-Gate Logic** — Noise-conditioned attention that adapts fusion strength
5. **Adaptive Noise Conditioning** — Model estimates and conditions on noise level dynamically
6. **Real-World Validation** — Works on PhysioNet PTB-XL (21,837 clinical records), not just synthetic
7. **Cross-Domain Transfer** — Train on ECG, test on audio/sensor (if it works → paper)
8. **Evaluation Framework** — Binned SNR analysis, spectral similarity, smoothness metrics, edge preservation

**Target venues:**
- NeurIPS (main track if results exceptional)
- ICML (main track)
- ICLR (strong fit — architecture novelty)
- IEEE TSP / Signal Processing Letters (journal)
- Nature Digital Medicine (if cardiac early warning added)

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | `--batch_size 32 --seq_len 512` |
| Loss not decreasing | Lower LR: `--lr 1e-4` |
| Over-smoothing warning | Increase spectral weight in train.py (0.25 → 0.35) |
| Real data not loading | Check path, run `pip install wfdb soundfile` |
| NaN in training | Reduce LR, check data normalization |

---

## 🔬 Citation

```bibtex
@software{adaptive_temporal_denoising_2026,
  title = {Adaptive Temporal Denoising Autoencoder},
  author = {Eric Yaka},
  year = {2026},
  url = {https://github.com/elbalor/temporal-signal-filter/tree/main/experiments/01_adaptive_temporal_denoising}
}
```

---

<div align="center">

**"Noise is just signal waiting to be understood"**

*Eric Yaka || The Digital Necromancer*

*Experiment 1 — Forged. Upgraded. Production-Ready.*

*Architecture: Multi-Scale TCN + Point-Wise Attention + Trinity Bottleneck + Strict Causality*

</div>
