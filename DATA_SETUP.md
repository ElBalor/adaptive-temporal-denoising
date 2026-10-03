# 📥 REAL DATA SETUP GUIDE

## Quick Start

This guide shows you how to download and use **real-world datasets** for training the adaptive denoising model.

---

## 🔥 WHY REAL DATA?

**Synthetic data** is great for prototyping, but **real data** is where models learn to handle:
- Actual noise patterns (not simulated)
- Patient variability (ECG)
- Real acoustic environments (audio)
- Human movement artifacts (sensor)

**Training on real data = Model that works in the real world.**

---

## 1️⃣ PHYSIONET PTB-XL (ECG)

### What It Is:
- **21,837** patient ECG records
- **12-lead** clinical-grade data
- Multiple diagnoses (MI, STTC, CD, HYP)

### Download Steps:

**Step 1: Create PhysioNet Account**
1. Go to https://physionet.org
2. Click "Register" (top right)
3. Fill out form (free, takes 2 min)
4. Verify email

**Step 2: Get Credentials**
- Your username = your email prefix
- You'll need this for wget

**Step 3: Download**
```bash
# Create data directory
mkdir -p data/physionet

# Download PTB-XL (replace YOUR_USER with your PhysioNet username)
wget -r -N -c -np --user YOUR_USER --ask-password \
    https://physionet.org/files/ptb-xl/1.0.3/ \
    -P data/physionet/
```

**Step 4: Verify**
```bash
ls data/physionet/ptb-xl/1.0.3/
# Should see: ptb-xl.csv, records folders, etc.
```

### Training Command:
```bash
python train.py \
    --epochs 100 \
    --batch_size 64 \
    --dataset ecg \
    --use_real \
    --real_ratio 0.3 \
    --real_data_dir ./data/physionet
```

---

## 2️⃣ LIBRISPEECH (Audio)

### What It Is:
- **1000 hours** of read English speech
- Clean and noisy subsets
- Standard ASR benchmark

### Download Steps:

**Step 1: Download**
```bash
# Create data directory
mkdir -p data/librispeech

# Download train-clean-100 (smallest subset, good for testing)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz \
    -P data/librispeech/

# Download test-clean (for evaluation)
wget https://www.openslr.org/resources/12/test-clean.tar.gz \
    -P data/librispeech/

# Extract
cd data/librispeech
tar xzvf train-clean-100.tar.gz
tar xzvf test-clean.tar.gz
```

**Step 2: Verify**
```bash
ls data/librispeech/LibriSpeech/train-clean-100/
# Should see: speaker ID folders
```

### Training Command:
```bash
python train.py \
    --epochs 100 \
    --batch_size 32 \
    --dataset audio \
    --use_real \
    --real_ratio 0.3 \
    --real_data_dir ./data/librispeech
```

---

## 3️⃣ UCI HAR (Sensor)

### What It Is:
- Accelerometer + gyroscope from **30 subjects**
- **6 activities** (walking, sitting, stairs, etc.)
- Smartphone-mounted sensors

### Download Steps:

**Step 1: Download**
```bash
# Create data directory
mkdir -p data/uci-har

# Download directly (no account needed)
wget https://archive.ics.uci.edu/static/public/123/human+activity+recognition+using+smartphones.zip \
    -P data/uci-har/

# Extract
cd data/uci-har
unzip human+activity+recognition+using+smartphones.zip
```

**Step 2: Verify**
```bash
ls UCI\ HAR\ Dataset/
# Should see: train/, test/, Inertial Signals/, etc.
```

### Training Command:
```bash
python train.py \
    --epochs 100 \
    --batch_size 64 \
    --dataset sensor \
    --use_real \
    --real_ratio 0.3 \
    --real_data_dir "./data/uci-har/UCI HAR Dataset"
```

---

## 🚀 FULL TRAINING WORKFLOW

### Option A: Start Fresh with Real Data
```bash
# Install dependencies
pip install wfdb soundfile librosa tonic

# Download your dataset(s)

# Train with mixed data
python train.py --epochs 100 --batch_size 64 --use_real --real_ratio 0.3 --real_data_dir ./data/physionet
```

### Option B: Add Real Data to Existing Training
```bash
# Resume from checkpoint + add real data
python train.py \
    --resume results/checkpoints/latest.pt \
    --epochs 100 \
    --use_real \
    --real_ratio 0.3 \
    --real_data_dir ./data/physionet
```

### Option C: Cross-Domain Transfer Test
```bash
# Train on ECG
python train.py --epochs 100 --dataset ecg --use_real --real_data_dir ./data/physionet

# Evaluate on Audio (tests generalization)
python evaluate.py \
    --checkpoint results/checkpoints/best.pt \
    --test-dataset audio
```

---

## 📊 EXPECTED IMPACT

| Metric | Synthetic Only | With Real Data |
|--------|---------------|----------------|
| **SNR** | 4-8 dB | **12-16 dB** |
| **Smoothness** | 0.2-0.4 (blurry) | **0.7-0.9** (sharp) |
| **Spectral Sim** | 0.85-0.90 | **0.92-0.96** |
| **Generalization** | Synthetic only | **Real-world data** |
| **Paper Potential** | Low | **High** |

---

## ⚠️ TROUBLESHOOTING

### "wfdb not installed"
```bash
pip install wfdb
```

### "PTB-XL not found"
- Check path: `ls data/physionet/ptb-xl/1.0.3/ptb-xl.csv`
- Make sure you used `--ask-password` for wget

### "Soundfile error"
```bash
pip install soundfile
```

### "CUDA out of memory"
- Reduce batch size: `--batch_size 32`
- Reduce sequence length: `--seq_len 512`

---

## 🎯 RECOMMENDED PATH

**For maximum impact:**

1. **Download PhysioNet PTB-XL first** (most impactful for ECG denoising)
2. **Train with 30% real data mix**
3. **Evaluate on both synthetic and real test sets**
4. **Compare results** — the improvement should be dramatic

---

<div align="center">

**"Noise is just signal waiting to be understood"**

*But only if your model sees real signals.*

</div>
