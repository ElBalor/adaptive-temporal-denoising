"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  VISUALIZATION SUITE                                                         ║
║  "Making the invisible visible"                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from pathlib import Path
import argparse
import json
from typing import Optional, List, Tuple
from scipy import signal as scipy_signal

from model import AdaptiveTemporalDenoisingAE
from data import AdaptiveDenoisingDataModule, NoiseInjector


# ──────────────────────────────────────────────────────────────────────────────
#  STYLE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

plt.style.use('dark_background')

COLORS = {
    'signal': '#00ff88',
    'noisy': '#ff6b6b',
    'denoised': '#4ecdc4',
    'error': '#ffe66d',
    'spectral': '#a855f7',
    'attention': '#f97316'
}

FONT_CONFIG = {
    'font.family': 'monospace',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
}

plt.rcParams.update(FONT_CONFIG)


# ──────────────────────────────────────────────────────────────────────────────
#  SIGNAL PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_signal_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    title: str = "Signal Denoising Comparison",
    save_path: Optional[Path] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Plot clean, noisy, and denoised signals in a comparison layout.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    t = np.arange(clean.shape[-1])
    
    # Noisy signal
    axes[0].plot(t, noisy[0, 0], color=COLORS['noisy'], linewidth=0.8, label='Noisy')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Noisy Input', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Denoised signal
    axes[1].plot(t, denoised[0, 0], color=COLORS['denoised'], linewidth=1.0, label='Denoised')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Denoised Output', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Clean signal (ground truth)
    axes[2].plot(t, clean[0, 0], color=COLORS['signal'], linewidth=0.8, label='Clean (GT)', alpha=0.7)
    axes[2].plot(t, denoised[0, 0], color=COLORS['denoised'], linewidth=1.0, label='Denoised', alpha=0.7)
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Comparison: Denoised vs Ground Truth', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    # Error / Residual
    error = clean - denoised
    axes[3].plot(t, error[0, 0], color=COLORS['error'], linewidth=0.8, label='Residual')
    axes[3].axhline(y=0, color='white', linestyle='--', linewidth=0.5)
    axes[3].set_xlabel('Time Step')
    axes[3].set_ylabel('Error')
    axes[3].set_title('Residual (Clean - Denoised)', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_spectral_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    sample_rate: float = 1.0,
    title: str = "Spectral Analysis",
    save_path: Optional[Path] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Plot frequency domain comparison using FFT.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Compute FFT
    n = clean.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1/sample_rate)
    
    clean_fft = np.fft.rfft(clean[0, 0], n=n)
    noisy_fft = np.fft.rfft(noisy[0, 0], n=n)
    denoised_fft = np.fft.rfft(denoised[0, 0], n=n)
    
    clean_mag = np.abs(clean_fft)
    noisy_mag = np.abs(noisy_fft)
    denoised_mag = np.abs(denoised_fft)
    
    # Noisy spectrum
    axes[0].plot(freqs, noisy_mag, color=COLORS['noisy'], linewidth=0.8, label='Noisy')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Noisy Spectrum', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Denoised spectrum
    axes[1].plot(freqs, denoised_mag, color=COLORS['denoised'], linewidth=1.0, label='Denoised')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Denoised Spectrum', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Overlay comparison
    axes[2].plot(freqs, clean_mag, color=COLORS['signal'], linewidth=0.8, label='Clean (GT)', alpha=0.7)
    axes[2].plot(freqs, denoised_mag, color=COLORS['denoised'], linewidth=1.0, label='Denoised', alpha=0.7)
    axes[2].set_xlabel('Frequency')
    axes[2].set_ylabel('Magnitude')
    axes[2].set_title('Spectral Comparison', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  ATTENTION VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_attention_weights(
    attention_weights: np.ndarray,
    noise_conditions: Optional[np.ndarray] = None,
    title: str = "Multi-Scale Attention",
    save_path: Optional[Path] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Visualize scale attention weights across samples.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    num_scales = attention_weights.shape[1]
    scale_names = [f'Scale {i+1}' for i in range(num_scales)]
    
    # Attention distribution histogram
    ax = axes[0]
    ax.hist(attention_weights, bins=20, alpha=0.6, label=scale_names, color=[COLORS['attention']] * num_scales)
    ax.set_xlabel('Attention Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Attention Weight Distribution', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Attention per sample (heatmap)
    ax = axes[1]
    im = ax.imshow(attention_weights[:100, :].T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Scale')
    ax.set_title('Attention Heatmap (First 100 Samples)', fontsize=10)
    ax.set_yticks(range(num_scales))
    ax.set_yticklabels(scale_names)
    plt.colorbar(im, ax=ax, label='Weight')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  TRAINING METRICS PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history_path: Path,
    save_path: Optional[Path] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Plot training and validation metrics from history JSON.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train = history['train']
    val = history['val']
    
    epochs = range(len(train))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')
    
    # Loss
    ax = axes[0, 0]
    train_loss = [h['loss'] for h in train]
    val_loss = [h['loss'] for h in val]
    ax.plot(epochs, train_loss, color=COLORS['signal'], linewidth=1.0, label='Train')
    ax.plot(epochs, val_loss, color=COLORS['noisy'], linewidth=1.0, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Over Time', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # SNR
    ax = axes[0, 1]
    train_snr = [h['snr'] for h in train]
    val_snr = [h['snr'] for h in val]
    ax.plot(epochs, train_snr, color=COLORS['signal'], linewidth=1.0, label='Train')
    ax.plot(epochs, val_snr, color=COLORS['noisy'], linewidth=1.0, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR Over Time', fontsize=10)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 0]
    lr = [h['lr'] for h in train]
    ax.plot(epochs, lr, color=COLORS['spectral'], linewidth=1.0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Loss components
    ax = axes[1, 1]
    train_mse = [h.get('mse', 0) for h in train]
    train_spectral = [h.get('spectral', 0) for h in train]
    ax.plot(epochs, train_mse, color=COLORS['signal'], linewidth=1.0, label='MSE')
    ax.plot(epochs, train_spectral, color=COLORS['spectral'], linewidth=1.0, label='Spectral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Component Value')
    ax.set_title('Loss Components (Train)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  LATENT SPACE VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_latent_space(
    latents: np.ndarray,
    noise_conditions: Optional[np.ndarray] = None,
    title: str = "Latent Space Analysis",
    save_path: Optional[Path] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Visualize latent space using PCA projection.
    """
    from sklearn.decomposition import PCA
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Flatten latents for PCA
    batch_size = latents.shape[0]
    latents_flat = latents.reshape(batch_size, -1)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents_flat)
    
    # Scatter plot
    ax = axes[0]
    scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                         c=noise_conditions if noise_conditions is not None else range(batch_size),
                         cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    ax.set_title('Latent Space PCA Projection', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Noise Level' if noise_conditions is not None else 'Sample')
    
    # Explained variance
    ax = axes[1]
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum) + 1), cumsum, color=COLORS['signal'], linewidth=1.5)
    ax.axhline(y=0.95, color=COLORS['noisy'], linestyle='--', label='95% variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA Explained Variance', fontsize=10)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  DENOISING DEMO
# ──────────────────────────────────────────────────────────────────────────────

def run_denoising_demo(
    checkpoint_path: Path,
    dataset_name: str = 'ecg',
    num_samples: int = 10,
    save_dir: Optional[Path] = None
):
    """
    Run denoising demo and generate all visualizations.
    """
    print("\n🎬 Running Denoising Demo...\n")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AdaptiveTemporalDenoisingAE(
        in_channels=1,
        hidden_channels=64,
        num_scales=3,
        num_encoder_blocks=4
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load data
    data_module = AdaptiveDenoisingDataModule(
        dataset_name=dataset_name,
        seq_len=1024,
        batch_size=num_samples,
        noise_config={'noise_type': 'mixed', 'snr_range': (5, 20)}
    )
    data_module.setup(train_size=100, val_size=num_samples, test_size=num_samples)
    
    test_loader = data_module.test_dataloader()
    noisy, clean, snrs = next(iter(test_loader))
    
    # Run inference
    with torch.no_grad():
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        output = model(noisy)
        denoised = output['output']
        attention = output['attention_weights'].cpu().numpy()
        latent = output['latent'].cpu().numpy()
        noise_cond = output['noise_condition'].cpu().numpy()
        
        noisy = noisy.cpu().numpy()
        clean = clean.cpu().numpy()
        denoised = denoised.cpu().numpy()
    
    # Compute metrics
    avg_snr = np.mean([10 * np.log10(
        np.mean(c ** 2) / np.mean((c - d) ** 2 + 1e-8)
    ) for c, d in zip(clean, denoised)])
    
    print(f"Average output SNR: {avg_snr:.2f} dB")
    
    # Create save directory
    if save_dir is None:
        save_dir = Path(__file__).parent / 'results' / 'figures'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...\n")
    
    # Signal comparison
    plot_signal_comparison(
        clean, noisy, denoised,
        title=f"Denoising Demo - {dataset_name.upper()} (Avg SNR: {avg_snr:.2f} dB)",
        save_path=save_dir / 'signal_comparison.png'
    )
    
    # Spectral analysis
    plot_spectral_comparison(
        clean, noisy, denoised,
        title=f"Spectral Analysis - {dataset_name.upper()}",
        save_path=save_dir / 'spectral_comparison.png'
    )
    
    # Attention weights
    plot_attention_weights(
        attention,
        title="Scale Attention Weights",
        save_path=save_dir / 'attention_weights.png'
    )
    
    # Latent space
    noise_norms = np.linalg.norm(noise_cond, axis=1)
    plot_latent_space(
        latent, noise_norms,
        title="Latent Space Analysis",
        save_path=save_dir / 'latent_space.png'
    )
    
    print(f"\n✅ All visualizations saved to: {save_dir}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize denoising results')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='ecg', choices=['ecg', 'audio', 'sensor'])
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='results/figures')
    parser.add_argument('--history', type=str, default=None, help='Path to training history JSON')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "VISUALIZATION SUITE" + " " * 33 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Run demo
    run_denoising_demo(
        checkpoint_path=checkpoint_path,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        save_dir=Path(args.save_dir)
    )
    
    # Plot history if available
    if args.history:
        history_path = Path(args.history)
        if history_path.exists():
            plot_training_history(
                history_path,
                save_path=Path(args.save_dir) / 'training_history.png'
            )
    
    print("🎨 Visualization complete!\n")


if __name__ == "__main__":
    main()
