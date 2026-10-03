"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  FULL SYSTEM TEST (MORPHIC BCI EDITION)                                      ║
║  "Verifying Causal Integrity and Trinity Logic"                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import numpy as np
from model import AdaptiveTemporalDenoisingAE, get_model_summary

def test_causal_integrity():
    """
    CRITICAL TEST: Ensures that y_t does NOT depend on x_{t+1}.
    If this fails, the model is 'cheating' by looking at the future.
    """
    print("🕒 Testing Causal Integrity...")
    
    model = AdaptiveTemporalDenoisingAE()
    model.eval()
    
    seq_len = 512
    x1 = torch.randn(1, 1, seq_len)
    
    # Create x2 where the second half is completely different
    x2 = x1.clone()
    x2[:, :, seq_len//2:] = torch.randn(1, 1, seq_len//2)
    
    with torch.no_grad():
        out1 = model(x1)['output']
        out2 = model(x2)['output']
    
    # The first half of the outputs should be IDENTICAL
    diff = torch.abs(out1[:, :, :seq_len//2] - out2[:, :, :seq_len//2]).max()
    
    print(f"   Max Difference in First Half: {diff.item():.2e}")
    
    if diff < 1e-5:
        print("   ✅ CAUSAL INTEGRITY PASSED: Present doesn't see the future.")
        return True
    else:
        print("   ❌ CAUSAL INTEGRITY FAILED: Future leakage detected!")
        return False

def test_pointwise_attention():
    """
    Verifies that attention weights vary over time.
    """
    print("🧠 Testing Point-wise Attention...")
    
    model = AdaptiveTemporalDenoisingAE()
    x = torch.randn(1, 1, 1024)
    
    with torch.no_grad():
        out = model(x)
        weights = out['attention_weights'] # [B, S, T]
        
    print(f"   Attention Weights Shape: {weights.shape}")
    
    # Check if weights vary across time
    variance = weights.var(dim=-1).mean()
    print(f"   Temporal Weight Variance: {variance.item():.2e}")
    
    if variance > 1e-6:
        print("   ✅ POINT-WISE ATTENTION PASSED: Model is morphing over time.")
        return True
    else:
        print("   ⚠️  LOW TEMPORAL VARIANCE: Model might be acting globally.")
        return False

def test_normalization():
    """
    Verifies the WeightNorm/RMSNorm hybrid strategy is stable.
    """
    print("⚡ Testing Normalization Stability...")
    
    model = AdaptiveTemporalDenoisingAE()
    
    # Feed a massive "voltage spike"
    x = torch.randn(1, 1, 512)
    x[:, :, 250:260] *= 100.0 
    
    with torch.no_grad():
        out = model(x)['output']
        
    # Check for NaNs and output range
    if torch.isnan(out).any():
        print("   ❌ STABILITY FAILED: NaNs detected after spike.")
        return False
    
    max_val = out.abs().max().item()
    print(f"   Max Output after Spike: {max_val:.4f}")
    
    if max_val <= 1.0: # Tanh bound
        print("   ✅ STABILITY PASSED: RMSNorm caught the spike.")
        return True
    else:
        print("   ❌ STABILITY FAILED: Output bounds exceeded.")
        return False

if __name__ == '__main__':
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "MORPHIC BCI SYSTEM VERIFICATION" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    results = [
        test_causal_integrity(),
        test_pointwise_attention(),
        test_normalization()
    ]
    
    if all(results):
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 18 + "🚀 ALL MORPHIC SYSTEMS GO — D3-READY" + " " * 23 + "║")
        print("╚" + "═" * 78 + "╝\n")
    else:
        print("\n❌ SYSTEM VERIFICATION FAILED. CHECK LOGS.\n")
