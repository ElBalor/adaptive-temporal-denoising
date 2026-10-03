# 🔬 EVALUATION PHILOSOPHY

## "The truth lies in the edge cases"

This document explains **what to watch** and **why it matters** when evaluating the denoising model.

---

## ⚠️ THE SILENT KILLERS

### 1. **Memorization vs Learning**

**The Trap:** Model performs well on synthetic data but fails on real variations.

**Detection:**
- Compare train vs test performance
- If gap > 3 dB → memorization
- If gap > 1 dB → moderate concern

**What to watch:**
```
Train Improvement: 12.5 dB
Test Improvement:  8.2 dB
Gap: 4.3 dB  ⚠️  MEMORIZATION detected!
```

**Fix:** More diverse training data, regularization, architectural constraints

---

### 2. **SNR Dependence**

**The Trap:** Model only works at high input SNR (15+ dB) but fails at 0-5 dB.

**Why it matters:** Real-world signals are often in the 0-10 dB regime.

**Detection:**
- Bin by input SNR: [0-5], [5-10], [10-15], [15-20], [20-30] dB
- Check improvement in each bin
- If 0-5 dB improvement < 2 dB → model is useless for hard cases

**What to watch:**
```
Input SNR Bin    Improvement
0-5 dB    →     0.8 dB   ⚠️  NOT WORKING where it matters
5-10 dB   →     3.2 dB   ⚠️  Weak
10-15 dB  →     6.5 dB   ✓   Okay
15-20 dB  →     9.1 dB   ✓   Good
20-30 dB  →     11.2 dB  ✓   Great
```

**The truth lives at 0-5 dB.** If it doesn't work there, it's not learning to denoise — it's learning to polish.

---

### 3. **Spectral Alignment (Identity Preservation)**

**The Trap:** Model "denoises" by smoothing everything into oblivion.

**Detection:**
- Compute spectral cosine similarity between clean and reconstructed
- If < 0.85 → model is distorting signal identity

**What to watch:**
```
Spectral Similarity: 0.72  ⚠️  LOW — signal identity lost
```

**Smoothness ratio:**
- Compute second derivative (roughness)
- Ratio = reconstructed_smoothness / clean_smoothness
- If < 0.5 → over-smoothing

```
Smoothness Ratio: 0.35  ⚠️  OVER-SMOOTHING — transients destroyed
```

---

### 4. **Edge/Transient Preservation**

**The Trap:** QRS complexes, spikes, onsets — the sharp bits that carry information — get blurred.

**Detection:**
- Find peaks in clean signal
- Check if peaks are preserved in reconstructed
- Correlation of local neighborhoods around peaks

**What to watch:**
```
Edge Preservation: 0.58  ⚠️  POOR — critical features being lost
```

**Why it matters:** In ECG, the QRS complex is diagnostic. In audio, transients define rhythm. Blurring them = destroying information.

---

### 5. **Noise Type Consistency**

**The Trap:** Model works on Gaussian noise but fails on impulse, colored, or real-world noise.

**Detection:**
- Test on each noise type separately: gaussian, poisson, impulse, colored
- Compute improvement variance across types

**What to watch:**
```
Noise Type    Improvement
Gaussian  →   10.2 dB  ✓
Poisson   →   8.5 dB   ✓
Impulse   →   2.1 dB   ⚠️  COLLAPSE
Colored   →   7.8 dB   ✓

Variance: 8.5  ⚠️  HIGH — model not robust
```

**Fix:** Train on mixed noise (already implemented), add more noise types

---

## 📊 THE EVALUATION SUITE

### Files:
- `evaluate.py` — Comprehensive post-training evaluation
- `train.py` — Now includes real-time monitoring during training

### Run Evaluation:
```bash
python evaluate.py --checkpoint results/checkpoints/best.pt
```

### Outputs:
- `results/evaluation/evaluation_report.txt` — Human-readable report
- `results/evaluation/evaluation_data.json` — Raw metrics
- `results/evaluation/snr_improvement_curve.png` — Visualization

---

## 🎯 WHAT GOOD LOOKS LIKE

### Healthy Model:
```
SNR Improvement vs Input SNR:
  0-5 dB:    8.5 ± 2.1 dB  ✓  Works in the hard cases
  5-10 dB:   9.2 ± 1.8 dB  ✓
  10-15 dB:  8.8 ± 1.5 dB  ✓
  15-20 dB:  7.5 ± 1.2 dB  ✓
  20-30 dB:  6.2 ± 1.0 dB  ✓

Spectral Alignment:
  Similarity:  0.94 ± 0.02  ✓  Identity preserved
  Smoothness:  0.92 ± 0.08  ✓  Not over-smoothing
  Edge Score:  0.89 ± 0.05  ✓  Transients intact

Noise Consistency:
  Gaussian:  8.5 dB
  Poisson:   8.2 dB
  Impulse:   7.9 dB
  Colored:   8.1 dB
  Variance:  0.06  ✓  Robust across types

Memorization Check:
  Train:  8.8 dB
  Test:   8.5 dB
  Gap:    0.3 dB  ✓  Generalizing, not memorizing
```

### Broken Model (what to avoid):
```
SNR Improvement vs Input SNR:
  0-5 dB:    0.5 ± 0.3 dB  ❌  USELESS where it matters
  5-10 dB:   2.1 ± 0.8 dB  ❌
  10-15 dB:  6.5 dB        ✓
  15-20 dB:  10.2 dB       ✓
  20-30 dB:  12.8 dB       ✓

Spectral Alignment:
  Similarity:  0.68  ❌  Signal identity destroyed
  Smoothness:  0.31  ❌  Over-smoothed
  Edge Score:  0.42  ❌  Transients obliterated

Noise Consistency:
  Gaussian:  11.2 dB
  Impulse:   1.5 dB   ❌  Collapse
  Variance:  12.5     ❌  Not robust

Memorization Check:
  Train:  11.5 dB
  Test:   6.2 dB
  Gap:    5.3 dB  ❌  MEMORIZATION
```

---

## 🔥 THE MINDSET

> "If your model performs too well on synthetic but drops on slight variations…
> **it's learning your generator, not reality.**"

This is the silent killer. The model can achieve 15 dB improvement on your test set and still be **completely useless** in the real world.

**The evaluation suite is your truth serum.** It tells you:
- Is it actually denoising, or just memorizing patterns?
- Does it work at 0-5 dB, or only on easy cases?
- Is it preserving signal identity, or smoothing everything?
- Does it generalize across noise types, or collapse?

---

## 🎯 WHAT TO DO WHEN TRAINING

1. **Watch TensorBoard live:**
   ```bash
   tensorboard --logdir results/logs
   ```
   
2. **Look for these curves:**
   - `val/spectral_similarity` — should increase, stay > 0.85
   - `val/smoothness_ratio` — should be ~0.8-1.2 (not < 0.5)
   - `val/low_snr_improvement` — should be > 3 dB

3. **Watch console output:**
   ```
   Epoch  45 | Spec Sim: 0.92 | Smooth: 0.88 | Low-SNR: 7.2 dB
   ```
   
   If you see warnings like:
   ```
   ⚠️  LOW spectral similarity: 0.72
   ⚠️  OVER-SMOOTHING: ratio=0.35
   ⚠️  POOR low-SNR performance: 1.2 dB
   ```
   
   → **Stop training. Something's wrong.**

4. **After training, run full evaluation:**
   ```bash
   python evaluate.py --checkpoint results/checkpoints/best.pt
   ```

---

## 🧠 THE DEEPER GAME

You're not training to get big numbers on a test set.

You're training a model that will face **signals it's never seen**, with **noise it's never encountered**, in **conditions you can't predict**.

The evaluation suite tests whether it's **actually learned the essence of denoising** — or just memorized your training distribution.

**Most models fail this test.**

Yours won't. Not if you watch the right metrics.

---

<div align="center">

**"Noise is just signal waiting to be understood"**

*But only if your model actually understands — not memorizes.*

</div>
