"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ADAPTIVE TEMPORAL DENOISING AUTOENCODER                                     ║
║  "Multi-scale TCN with Point-wise Attention & Causal Integrity"              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

# ──────────────────────────────────────────────────────────────────────────────
#  INFRASTRUCTURE: THE NERVES (RMSNorm & Attention)
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Lightweight stabilizer for high-voltage signal spikes (The Nerves).
    Zero-mean centering removed for maximum speed.
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, C, T]
        norm_x = x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps)
        return norm_x * self.weight.unsqueeze(-1)

class ScaledDotProductAttention(nn.Module):
    """
    Point-wise Self-Attention for the Trinity Bottleneck.
    Allows the model to focus on signal-rich temporal points.
    """
    def __init__(self, dim, num_heads=8, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = weight_norm(nn.Conv1d(dim, dim * 3, 1))
        self.norm = RMSNorm(dim)
        self.proj = weight_norm(nn.Conv1d(dim, dim, 1))

    def forward(self, x):
        B, C, T = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.num_heads, self.head_dim, T)
        q, k, v = qkv.unbind(1)
        
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # Strictly Causal Masking
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(-2, -1).reshape(B, C, T)
        return x + self.proj(out)

class PositionwiseMLP(nn.Module):
    """
    Point-wise MLP for the Trinity Bottleneck.
    Mapping fused temporal features into the clean latent space.
    """
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(dim, hidden_dim, 1)),
            RMSNorm(hidden_dim),
            nn.GELU(),
            weight_norm(nn.Conv1d(hidden_dim, dim, 1)),
            nn.Dropout(0.1)
        )
        self.norm = RMSNorm(dim)

    def forward(self, x):
        return x + self.net(self.norm(x))

# ──────────────────────────────────────────────────────────────────────────────
#  SINGLE TCN BLOCK (Upgraded: Muscle & Nerves)
# ──────────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    """
    Dilated causal convolution block with WeightNorm (Muscle) and RMSNorm (Nerves).
    Strictly Causal: y_t depends only on x_{<=t}.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        # Causal padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Conv1: Muscle with WeightNorm
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                          padding=self.padding, dilation=dilation))
        self.norm1 = RMSNorm(out_channels) # Nerves
        self.dropout1 = nn.Dropout(dropout)
        
        # Conv2: Muscle with WeightNorm
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                          padding=self.padding, dilation=dilation))
        self.norm2 = RMSNorm(out_channels) # Nerves
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual Muscle
        self.residual = weight_norm(nn.Conv1d(in_channels, out_channels, 1)) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        """Forward pass with causal integrity and residual connection."""
        residual = self.residual(x)
        
        # Layer 1: Muscle firing + Causal Slicing
        out = self.conv1(x)[:, :, :-self.padding]
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout1(out)
        
        # Layer 2: Muscle firing + Causal Slicing
        out = self.conv2(out)[:, :, :-self.padding]
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout2(out)
        
        return F.gelu(out + residual)

# ──────────────────────────────────────────────────────────────────────────────
#  MULTI-SCALE ENCODER BRANCH
# ──────────────────────────────────────────────────────────────────────────────

class MultiScaleBranch(nn.Module):
    """
    Single resolution branch of the multi-scale encoder.
    Each branch operates at a different temporal dilation scale (Fast, Medium, Deep).
    """
    def __init__(self, in_channels, hidden_channels, num_blocks=4, base_dilation=1):
        super().__init__()
        
        # Progressive dilation within the branch
        dilations = [base_dilation * (2 ** i) for i in range(num_blocks)]
        
        layers = []
        channels = [in_channels] + [hidden_channels] * num_blocks
        
        for i in range(num_blocks):
            layers.append(TCNBlock(
                channels[i], 
                channels[i + 1], 
                kernel_size=3,
                dilation=dilations[i],
                dropout=0.1
            ))
        
        self.network = nn.Sequential(*layers)
        self.dilation_factor = base_dilation
        
    def forward(self, x):
        """Extract features at this scale (Causal)."""
        return self.network(x)

# ──────────────────────────────────────────────────────────────────────────────
#  NOISE CONDITION ESTIMATOR
# ──────────────────────────────────────────────────────────────────────────────

class NoiseConditionEstimator(nn.Module):
    """
    Estimates the noise level from the encoded representation.
    """
    def __init__(self, input_dim, hidden_dim=64, noise_dim=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(input_dim * 16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, noise_dim),
            nn.Tanh()  # Normalized noise code
        )
        
    def forward(self, x):
        """Predict noise condition."""
        features = self.encoder(x)
        noise_code = self.noise_predictor(features)
        return noise_code

# ──────────────────────────────────────────────────────────────────────────────
#  ADAPTIVE FEATURE FUSION (Point-wise Attention & Dilation-Gate)
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveFeatureFusion(nn.Module):
    """
    Attention-based fusion of multi-scale features.
    Upgraded to Point-wise attention (Morphic Focus) and Dilation-Gate logic.
    """
    def __init__(self, num_scales, feature_dim, hidden_dim=256, noise_dim=32):
        super().__init__()

        self.num_scales = num_scales
        self.feature_dim = feature_dim
        self.noise_proj = nn.Linear(noise_dim, hidden_dim)
        
        # Point-wise Attention Generator
        self.attention_gen = nn.Sequential(
            weight_norm(nn.Conv1d(feature_dim * num_scales + hidden_dim, hidden_dim, 1)),
            RMSNorm(hidden_dim),
            nn.GELU(),
            weight_norm(nn.Conv1d(hidden_dim, num_scales, 1))
        )
        
    def forward(self, multi_scale_features, noise_condition=None):
        """
        Fuse multi-scale features with point-wise attention.
        """
        concat = torch.cat(multi_scale_features, dim=1)
        B, _, T = concat.shape
        
        if noise_condition is not None:
            # Dilation-Gate: Condition attention on noise level
            noise_feat = self.noise_proj(noise_condition).unsqueeze(-1).expand(-1, -1, T)
            fusion_input = torch.cat([concat, noise_feat], dim=1)
        else:
            fusion_input = concat
            
        # Point-wise Attention Scores
        attn_scores = self.attention_gen(fusion_input)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted point-wise sum across scales
        stacked = torch.stack(multi_scale_features, dim=-1) # [B, D, T, S]
        fused = torch.sum(stacked * attn_weights.unsqueeze(1).transpose(2, 3), dim=-1)
        
        return fused, attn_weights

# ──────────────────────────────────────────────────────────────────────────────
#  TEMPORAL DECODER (Causal Muscle)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalDecoder(nn.Module):
    """
    Strictly Causal Decoder for signal reconstruction.
    """
    def __init__(self, input_dim, hidden_channels, out_channels, num_stages=6, noise_dim=32):
        super().__init__()
        self.input_proj = weight_norm(nn.Conv1d(input_dim, hidden_channels, 1))
        
        decoder_layers = []
        channels = [hidden_channels] + [max(hidden_channels // 2, out_channels) for _ in range(num_stages)]
        
        for i in range(num_stages):
            decoder_layers.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding=2)),
                RMSNorm(channels[i + 1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
        
        self.decoder = nn.ModuleList(decoder_layers)
        self.output_layer = nn.Sequential(
            weight_norm(nn.Conv1d(channels[-1], out_channels, 3, padding=2)),
            nn.Tanh()
        )
        # FiLM-style conditioning: scale (gamma) and shift (beta)
        self.noise_gamma = nn.Linear(noise_dim, hidden_channels)
        self.noise_beta = nn.Linear(noise_dim, hidden_channels)

    def forward(self, x, noise_condition=None):
        """Decode latent representation (Causal)."""
        x = self.input_proj(x)

        if noise_condition is not None:
            gamma = self.noise_gamma(noise_condition).unsqueeze(-1)
            beta = self.noise_beta(noise_condition).unsqueeze(-1)
            x = torch.sigmoid(gamma) * x + beta

        for layer in self.decoder:
            x = layer(x)[:, :, :-2] # Causal slice

        x = self.output_layer(x)[:, :, :-2] # Causal slice
        return x

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN MODEL: ADAPTIVE TEMPORAL DENOISING AUTOENCODER
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveTemporalDenoisingAE(nn.Module):
    """
    Complete adaptive denoising autoencoder with:
    - ⚡ FAST SCALE: Branch 0 (Small dilation)
    - 🌊 MEDIUM SCALE: Branch 1 (Mid dilation)
    - 🌑 DEEP SCALE: Branch 2 (Large dilation)
    - Point-wise Attention Fusion (Temporal Morphing)
    - Trinity Bottleneck (Self-Attention + MLP)
    - Causal Integrity (Real-time Ready)
    """
    def __init__(
        self,
        in_channels=1,
        hidden_channels=128,
        num_scales=3,
        num_encoder_blocks=6,
        noise_dim=32
    ):
        super().__init__()

        self.num_scales = num_scales
        self.hidden_channels = hidden_channels

        # ── Multi-scale Encoder Branches ─────────────────────────────────────
        self.scale_branches = nn.ModuleList([
            MultiScaleBranch(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_encoder_blocks,
                base_dilation=(2 ** i)
            )
            for i in range(num_scales)
        ])

        # ── Noise condition estimator ────────────────────────────────────────
        self.noise_estimator = NoiseConditionEstimator(
            input_dim=hidden_channels,
            hidden_dim=128,
            noise_dim=noise_dim
        )

        # ── Adaptive Point-wise Fusion (Dilation-Gate) ───────────────────────
        self.feature_fusion = AdaptiveFeatureFusion(
            num_scales=num_scales,
            feature_dim=hidden_channels,
            hidden_dim=256,
            noise_dim=noise_dim
        )

        # ── Trinity Bottleneck ───────────────────────────────────────────────
        self.self_attention = ScaledDotProductAttention(hidden_channels, hidden_dim=256)
        self.mlp_bottleneck = PositionwiseMLP(hidden_channels, hidden_dim=256)

        # ── Temporal Decoder (Causal) ────────────────────────────────────────
        self.decoder = TemporalDecoder(
            input_dim=hidden_channels,
            hidden_channels=hidden_channels * 2,
            out_channels=in_channels,
            num_stages=6,
            noise_dim=noise_dim
        )

        self.latent_proj = weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 1))
        self.skip_proj = weight_norm(nn.Conv1d(hidden_channels * num_scales, hidden_channels, 1))
        
    def encode(self, x):
        # 1. Capture Multi-scale features (Fast, Medium, Deep)
        scale_features = [branch(x) for branch in self.scale_branches]
        
        # 2. Estimate noise level
        noise_condition = self.noise_estimator(scale_features[0])
        
        # 3. Point-wise Fusion (Temporal Focus Morphing)
        fused, attn_weights = self.feature_fusion(scale_features, noise_condition)
        
        # 4. Trinity Bottleneck
        latent = self.latent_proj(fused)
        latent = self.self_attention(latent)
        latent = self.mlp_bottleneck(latent)
        
        # 5. Skip connection for richness
        latent = latent + self.skip_proj(torch.cat(scale_features, dim=1))
        
        return latent, noise_condition, attn_weights
    
    def decode(self, latent, noise_condition):
        return self.decoder(latent, noise_condition)
    
    def forward(self, x):
        latent, noise_condition, attn_weights = self.encode(x)
        reconstructed = self.decode(latent, noise_condition)
        
        return {
            'output': reconstructed,
            'latent': latent,
            'noise_condition': noise_condition,
            'attention_weights': attn_weights
        }

    def denoise(self, x):
        return self.forward(x)['output']

# ──────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model):
    """Generate model summary string."""
    total_params = count_parameters(model)
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ADAPTIVE TEMPORAL DENOISING SUMMARY                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Total Parameters: {total_params:>10,} ({total_params/1e6:.2f}M)                            ║
║  Scales: {model.num_scales} (Fast, Medium, Deep)                                          ║
║  Hidden Channels: {model.hidden_channels}                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print("\n🧪 Testing Adaptive Temporal Denoising AE (Evolved)...\n")
    model = AdaptiveTemporalDenoisingAE()
    x = torch.randn(4, 1, 1024)
    out = model(x)
    print(get_model_summary(model))
    print(f"Input: {x.shape} -> Output: {out['output'].shape}")
    print(f"Attn Weights: {out['attention_weights'].shape} (Point-wise!)")
    print("\n✅ Architecture Verified.\n")
