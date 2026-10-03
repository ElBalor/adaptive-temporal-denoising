"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TRAINING LOOP & LOGGING                                                     ║
║  "Where gradients meet chaos"                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import json
from tqdm import tqdm

import sys
from pathlib import Path
# Add experiments root to sys.path for telemetry import
sys.path.append(str(Path(__file__).parent.parent))
from telemetry_client import TelemetryClient

# ── CUDA Speed Optimizations ─────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True          # Auto-tune conv algorithms
torch.backends.cudnn.allow_tf32 = True         # TensorFloat-32 (10-30% speedup)
torch.set_float32_matmul_precision('high')     # Enable TF32 matmul

# Import from small-* files using importlib (hyphenated names)
import importlib.util
spec_model = importlib.util.spec_from_file_location("model_small", str(Path(__file__).parent / "model-small.py"))
model_small = importlib.util.module_from_spec(spec_model)
spec_model.loader.exec_module(model_small)
AdaptiveTemporalDenoisingAE = model_small.AdaptiveTemporalDenoisingAE
get_model_summary = model_small.get_model_summary

spec_data = importlib.util.spec_from_file_location("data_small", str(Path(__file__).parent / "data-small.py"))
data_small = importlib.util.module_from_spec(spec_data)
spec_data.loader.exec_module(data_small)
AdaptiveDenoisingDataModule = data_small.AdaptiveDenoisingDataModule
compute_snr = data_small.compute_snr
compute_sdr = data_small.compute_sdr
compute_snr_batch = data_small.compute_snr_batch
compute_sdr_batch = data_small.compute_sdr_batch


# ──────────────────────────────────────────────────────────────────────────────
#  ADVANCED METRICS (for detecting failure modes)
# ──────────────────────────────────────────────────────────────────────────────

def compute_spectral_similarity(clean: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Spectral cosine similarity — detects if signal identity is preserved.
    Returns value in [0, 1] where 1 = identical spectra.
    """
    clean_fft = torch.fft.rfft(clean, dim=-1).abs()
    recon_fft = torch.fft.rfft(reconstructed, dim=-1).abs()
    
    # Flatten for batch-wise cosine similarity
    clean_flat = clean_fft.view(clean_fft.shape[0], -1)
    recon_flat = recon_fft.view(recon_fft.shape[0], -1)
    
    similarity = torch.nn.functional.cosine_similarity(clean_flat, recon_flat, dim=-1)
    return similarity


def compute_smoothness_ratio(clean: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Ratio of smoothness (second derivative).
    < 1.0 means over-smoothing (reconstructed is smoother than clean).
    """
    # Second derivative
    clean_diff2 = torch.diff(clean, n=2, dim=-1)
    recon_diff2 = torch.diff(reconstructed, n=2, dim=-1)
    
    clean_smooth = torch.mean(clean_diff2 ** 2, dim=-1)
    recon_smooth = torch.mean(recon_diff2 ** 2, dim=-1)
    
    return recon_smooth / (clean_smooth + 1e-8)


def compute_snr_by_bin(input_snrs: torch.Tensor, output_snrs: torch.Tensor, 
                       bins: List[Tuple[float, float]]) -> Dict[str, torch.Tensor]:
    """
    Compute SNR improvement binned by input SNR.
    Critical for detecting if model only works at high SNR.
    """
    bin_results = {}
    
    for low, high in bins:
        mask = (input_snrs >= low) & (input_snrs < high)
        if mask.any():
            in_mean = input_snrs[mask].mean()
            out_mean = output_snrs[mask].mean()
            improvement = (output_snrs[mask] - input_snrs[mask]).mean()
            bin_results[f"{low}-{high}dB"] = {
                'input': in_mean.item(),
                'output': out_mean.item(),
                'improvement': improvement.item(),
                'count': mask.sum().item()
            }
    
    return bin_results

class CompositeDenoisingLoss(nn.Module):
    """
    Multi-objective loss for robust denoising:
    - MSE reconstruction (weighted lower to prevent over-smoothing)
    - Spectral loss (weighted higher to preserve frequency structure)
    - SSIM loss (structural similarity — directly prevents over-smoothing)
    - Latent regularization

    Loss balance designed to prevent the "blur loophole" where models
    minimize MSE by averaging instead of intelligent denoising.
    """

    def __init__(
        self,
        mse_weight: float = 0.5,      # Reduced: prevents over-smoothing incentive
        spectral_weight: float = 0.25,  # Increased: forces frequency preservation
        ssim_weight: float = 0.2,      # NEW: structural similarity
        latent_weight: float = 0.01
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.ssim_weight = ssim_weight
        self.latent_weight = latent_weight

        self.mse = nn.MSELoss()
        
    def spectral_loss(self, clean: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Complex spectral loss — handles phase correctly via complex distance.
        
        Uses log-magnitude for perceptual relevance + complex L1 for proper phase handling.
        """
        clean_fft = torch.fft.rfft(clean, dim=-1)
        recon_fft = torch.fft.rfft(reconstructed, dim=-1)
        
        # Log-magnitude loss (perceptually meaningful, handles dynamic range)
        mag_loss = F.l1_loss(
            torch.log(clean_fft.abs() + 1e-8),
            torch.log(recon_fft.abs() + 1e-8)
        )
        
        # Complex spectral loss (handles phase correctly — no circular issues)
        complex_loss = F.l1_loss(clean_fft, recon_fft)
        
        return mag_loss + complex_loss

    def ssim_loss(self, clean: torch.Tensor, reconstructed: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """
        1D Structural Similarity Index loss.
        
        SSIM measures structural similarity by comparing:
        - Luminance (mean)
        - Contrast (variance)
        - Structure (covariance)
        
        Returns 1 - SSIM (since we minimize).
        """
        # Constants for numerical stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Ensure signals are [batch, 1, length] for conv1d
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)
        if reconstructed.dim() == 2:
            reconstructed = reconstructed.unsqueeze(1)
        
        # Create 1D Gaussian window
        coords = torch.arange(window_size, device=clean.device).float() - window_size // 2
        gaussian = torch.exp(-coords ** 2 / (2 * (window_size / 4) ** 2))
        gaussian = gaussian / gaussian.sum()
        window = gaussian.view(1, 1, -1).repeat(clean.shape[1], 1, 1)
        
        # Compute local statistics
        mu1 = F.conv1d(clean, window, padding=window_size // 2, groups=clean.shape[1])
        mu2 = F.conv1d(reconstructed, window, padding=window_size // 2, groups=reconstructed.shape[1])
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv1d(clean * clean, window, padding=window_size // 2, groups=clean.shape[1]) - mu1_sq
        sigma2_sq = F.conv1d(reconstructed * reconstructed, window, padding=window_size // 2, groups=reconstructed.shape[1]) - mu2_sq
        sigma12 = F.conv1d(clean * reconstructed, window, padding=window_size // 2, groups=clean.shape[1]) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return mean SSIM loss
        return 1.0 - ssim_map.mean()
    
    def forward(
        self,
        clean: torch.Tensor,
        reconstructed: torch.Tensor,
        latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss.
        
        Returns:
            total_loss, loss_components dict
        """
        # Reconstruction MSE
        mse_loss = self.mse(reconstructed, clean)

        # Spectral loss
        spec_loss = self.spectral_loss(clean, reconstructed)

        # SSIM loss (structural similarity)
        ssim_loss = self.ssim_loss(clean, reconstructed)

        # Latent regularization (encourage small magnitudes)
        if latent is not None:
            latent_loss = torch.mean(latent ** 2)
        else:
            latent_loss = torch.tensor(0.0, device=clean.device)

        # Total
        total = (
            self.mse_weight * mse_loss +
            self.spectral_weight * spec_loss +
            self.ssim_weight * ssim_loss +
            self.latent_weight * latent_loss
        )

        components = {
            'total': total.item(),
            'mse': mse_loss.item(),
            'spectral': spec_loss.item(),
            'ssim': ssim_loss.item(),
            'latent': latent_loss.item()
        }
        
        return total, components


# ──────────────────────────────────────────────────────────────────────────────
#  METRICS TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Track and log training metrics."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.writer = SummaryWriter(str(log_dir))
        self.history = {
            'train': [],
            'val': []
        }
        self.best_val_loss = float('inf')
        self.best_val_snr = 0
        
    def log_train(
        self,
        epoch: int,
        loss: float,
        snr: float,
        lr: float,
        loss_components: Dict[str, float]
    ):
        """Log training metrics."""
        self.writer.add_scalar('train/loss', loss, epoch)
        self.writer.add_scalar('train/snr', snr, epoch)
        self.writer.add_scalar('train/lr', lr, epoch)
        
        for name, value in loss_components.items():
            self.writer.add_scalar(f'train/{name}', value, epoch)
        
        self.history['train'].append({
            'epoch': epoch,
            'loss': loss,
            'snr': snr,
            'lr': lr,
            **loss_components
        })
    
    def log_val(
        self,
        epoch: int,
        loss: float,
        snr: float,
        sdr: float,
        loss_components: Dict[str, float],
        advanced_metrics: Optional[Dict] = None
    ):
        """Log validation metrics."""
        self.writer.add_scalar('val/loss', loss, epoch)
        self.writer.add_scalar('val/snr', snr, epoch)
        self.writer.add_scalar('val/sdr', sdr, epoch)

        for name, value in loss_components.items():
            self.writer.add_scalar(f'val/{name}', value, epoch)

        val_entry = {
            'epoch': epoch,
            'loss': loss,
            'snr': snr,
            'sdr': sdr,
            **loss_components
        }
        
        # Add advanced metrics if available
        if advanced_metrics:
            val_entry['spectral_similarity'] = advanced_metrics['spectral_similarity']
            val_entry['smoothness_ratio'] = advanced_metrics['smoothness_ratio']
            val_entry['snr_by_bin'] = advanced_metrics.get('snr_by_bin', {})
            val_entry['low_snr_improvement'] = advanced_metrics.get('low_snr_improvement')
            val_entry['warnings'] = advanced_metrics.get('warnings', [])
        
        self.history['val'].append(val_entry)

        # Track best
        if loss < self.best_val_loss:
            self.best_val_loss = loss
        if snr > self.best_val_snr:
            self.best_val_snr = snr
    
    def save_history(self):
        """Save training history to JSON (with tensor conversion)."""
        def convert_tensors(obj):
            """Recursively convert tensors to Python scalars."""
            if isinstance(obj, torch.Tensor):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(v) for v in obj]
            return obj
        
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(convert_tensors(self.history), f, indent=2)
    
    def close(self):
        """Close tensorboard writer."""
        self.writer.close()


# ──────────────────────────────────────────────────────────────────────────────
#  TRAINER
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveDenoisingTrainer:
    """
    Complete training pipeline for adaptive denoising autoencoder.
    """
    
    def __init__(
        self,
        model: AdaptiveTemporalDenoisingAE,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        self.scheduler = scheduler or optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        self.criterion = CompositeDenoisingLoss(
            mse_weight=0.5,
            spectral_weight=0.4,
            ssim_weight=0.4,
            latent_weight=0.01
        )
        
        # Logging
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent / 'results' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(__file__).parent / 'results' / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = MetricsTracker(self.log_dir)
        
        # Telemetry for 3D Dashboard
        self.telemetry = TelemetryClient("Temporal_Denoising")
        
        print(f"\n🔥 Training on: {self.device.upper()}")
        print(get_model_summary(model))
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_snr = 0
        all_components = {'mse': 0, 'spectral': 0, 'latent': 0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [TRAIN]')
        
        for noisy, clean, snrs in pbar:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(noisy)
            
            reconstructed = output['output']
            latent = output['latent']
            
            # Compute loss
            loss, components = self.criterion(clean, reconstructed, latent)

            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            # Compute SNR (per-sample, then average)
            with torch.no_grad():
                out_snr = compute_snr_batch(clean, reconstructed)

            # Accumulate
            total_loss += loss.item()
            total_snr += out_snr
            for key in all_components:
                all_components[key] += components.get(key, 0)
            num_batches += 1
            
            # Send real-time telemetry
            if num_batches % 10 == 0:
                self.telemetry.log_metrics(epoch, {
                    "accuracy": out_snr / 40.0, # Approximate scale for accuracy bar
                    "loss": loss.item()
                })
                # Stream first sample in batch
                self.telemetry.stream_signal(
                    input_sig=noisy[0, 0].detach().cpu().numpy(),
                    output_sig=reconstructed[0, 0].detach().cpu().numpy()
                )
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'snr': f'{out_snr:.2f} dB'
            })
        
        avg_loss = total_loss / num_batches
        avg_snr = total_snr / num_batches
        avg_components = {k: v / num_batches for k, v in all_components.items()}
        
        return avg_loss, avg_components, avg_snr
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict, float, float, Dict]:
        """
        Validate on validation set with advanced failure mode detection.
        
        Returns:
            avg_loss, avg_components, avg_snr, avg_sdr, advanced_metrics
        """
        self.model.eval()

        total_loss = 0
        total_snr = 0
        total_sdr = 0
        all_components = {'mse': 0, 'spectral': 0, 'latent': 0}
        num_batches = 0
        
        # Advanced metrics accumulators
        all_input_snrs = []
        all_output_snrs = []
        spectral_sims = []
        smoothness_ratios = []

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [VAL]  ')

        for noisy, clean, snrs in pbar:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # Forward pass
            output = self.model(noisy)
            reconstructed = output['output']
            latent = output['latent']

            # Compute loss
            loss, components = self.criterion(clean, reconstructed, latent)

            # Compute metrics (per-sample, then average)
            out_snr = compute_snr_batch(clean, reconstructed)
            out_sdr = compute_sdr_batch(clean, reconstructed)
            
            # Advanced metrics (unsqueeze to make 1D for concatenation)
            input_snrs = compute_snr_batch(clean, noisy).unsqueeze(0)
            out_snr_unsq = out_snr.unsqueeze(0)
            spectral_sim = compute_spectral_similarity(clean, reconstructed)
            smooth_ratio = compute_smoothness_ratio(clean, reconstructed)
            
            all_input_snrs.append(input_snrs)
            all_output_snrs.append(out_snr_unsq)
            spectral_sims.append(spectral_sim)
            smoothness_ratios.append(smooth_ratio)

            # Accumulate
            total_loss += loss.item()
            total_snr += out_snr
            total_sdr += out_sdr
            for key in all_components:
                all_components[key] += components.get(key, 0)
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'snr': f'{out_snr:.2f} dB',
                'sdr': f'{out_sdr:.2f} dB'
            })

        avg_loss = total_loss / num_batches
        avg_snr = total_snr / num_batches
        avg_sdr = total_sdr / num_batches
        avg_components = {k: v / num_batches for k, v in all_components.items()}
        
        # Compute advanced metrics
        input_snrs_all = torch.cat(all_input_snrs)
        output_snrs_all = torch.cat(all_output_snrs)
        spectral_sims_all = torch.cat(spectral_sims)
        smoothness_ratios_all = torch.cat(smoothness_ratios)
        
        # SNR bins for edge case detection
        snr_bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
        snr_by_bin = compute_snr_by_bin(input_snrs_all, output_snrs_all, snr_bins)
        
        advanced_metrics = {
            'spectral_similarity': spectral_sims_all.mean().item(),
            'smoothness_ratio': smoothness_ratios_all.mean().item(),
            'snr_by_bin': snr_by_bin,
            'low_snr_improvement': snr_by_bin.get('0-5dB', {}).get('improvement', None),
        }
        
        # Warnings
        warnings = []
        if advanced_metrics['spectral_similarity'] < 0.85:
            warnings.append(f"LOW spectral similarity: {advanced_metrics['spectral_similarity']:.3f}")
        if advanced_metrics['smoothness_ratio'] < 0.5:
            warnings.append(f"OVER-SMOOTHING: ratio={advanced_metrics['smoothness_ratio']:.3f}")
        if advanced_metrics.get('low_snr_improvement') is not None and advanced_metrics['low_snr_improvement'] < 2.0:
            warnings.append(f"POOR low-SNR performance: {advanced_metrics['low_snr_improvement']:.2f} dB")
        
        advanced_metrics['warnings'] = warnings

        return avg_loss, avg_components, avg_snr, avg_sdr, advanced_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': self.metrics.history
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
    
    def train(
        self,
        num_epochs: int = 100,
        resume_from: Optional[str] = None
    ):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        start_epoch = 0
        
        # Resume from checkpoint
        if resume_from:
            print(f"\n📥 Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resumed from epoch {start_epoch}")
        
        print(f"\n🚀 Starting training for {num_epochs} epochs...\n")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_components, train_snr = self.train_epoch(epoch)

            # Validate (with advanced failure mode detection)
            val_loss, val_components, val_snr, val_sdr, advanced_metrics = self.validate(epoch)

            # Learning rate step
            if self.scheduler:
                self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]['lr']

            self.metrics.log_train(
                epoch, train_loss, train_snr, lr, train_components
            )
            
            # Log advanced metrics to tensorboard
            self.metrics.writer.add_scalar(
                'val/spectral_similarity',
                advanced_metrics['spectral_similarity'],
                epoch
            )
            self.metrics.writer.add_scalar(
                'val/smoothness_ratio',
                advanced_metrics['smoothness_ratio'],
                epoch
            )
            if advanced_metrics.get('low_snr_improvement') is not None:
                self.metrics.writer.add_scalar(
                    'val/low_snr_improvement',
                    advanced_metrics['low_snr_improvement'],
                    epoch
                )

            self.metrics.log_val(
                epoch, val_loss, val_snr, val_sdr, val_components, advanced_metrics
            )

            # Checkpoint
            is_best = val_loss < self.metrics.best_val_loss
            self.save_checkpoint(epoch, is_best)

            # Progress summary
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            
            # Build warning message if any
            warning_str = ""
            if advanced_metrics.get('warnings'):
                warning_str = " | ⚠️ " + " | ".join(advanced_metrics['warnings'])

            print("\n" + "=" * 80)
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val SNR: {val_snr:.2f} dB | "
                  f"Val SDR: {val_sdr:.2f} dB")
            print(f"Spec Sim: {advanced_metrics['spectral_similarity']:.3f} | "
                  f"Smooth: {advanced_metrics['smoothness_ratio']:.3f} |"
                  f"Low-SNR: {advanced_metrics.get('low_snr_improvement', 'N/A')}" +
                  (f" dB" if advanced_metrics.get('low_snr_improvement') is not None else ""))
            print(f"Time: {epoch_time:.1f}s | "
                  f"Total: {elapsed/3600:.2f}h | "
                  f"Best Val Loss: {self.metrics.best_val_loss:.4f} | "
                  f"Best Val SNR: {self.metrics.best_val_snr:.2f} dB" +
                  warning_str)
            print("=" * 80 + "\n")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🏁 Training complete!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.metrics.best_val_loss:.4f}")
        print(f"Best validation SNR: {self.metrics.best_val_snr:.2f} dB")
        
        # Save final history
        self.metrics.save_history()
        self.metrics.close()
        
        return self.metrics.history


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train Adaptive Temporal Denoising Autoencoder')

    # Data
    parser.add_argument('--dataset', type=str, default='ecg', choices=['ecg', 'audio', 'sensor'])
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--noise_type', type=str, default='mixed', choices=['gaussian', 'poisson', 'impulse', 'colored', 'mixed'])
    parser.add_argument('--snr_min', type=float, default=5.0)
    parser.add_argument('--snr_max', type=float, default=30.0)
    
    # Real data options
    parser.add_argument('--use_real', action='store_true', help='Use real datasets (PhysioNet, LibriSpeech, etc.)')
    parser.add_argument('--real_ratio', type=float, default=0.3, help='Ratio of real data in mixed training')
    parser.add_argument('--real_data_dir', type=str, default=None, help='Path to real data directory')

    # Model
    parser.add_argument('--hidden_channels', type=int, default=64)  # Smaller for quick test
    parser.add_argument('--num_scales', type=int, default=2)  # Fewer scales
    parser.add_argument('--num_encoder_blocks', type=int, default=3)  # Fewer blocks

    # Training
    parser.add_argument('--epochs', type=int, default=5)  # Very few epochs for convergence test
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')

    # Logging
    parser.add_argument('--log_dir', type=str, default='results/logs')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints')

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "ADAPTIVE TEMPORAL DENOISING" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    
    if args.use_real:
        print("🔥 REAL DATA MODE: Mixing synthetic + real data")
        print(f"   Real ratio: {args.real_ratio * 100:.0f}%")
        print(f"   Real data dir: {args.real_data_dir or 'Not specified'}")
    else:
        print("📊 SYNTHETIC MODE: Training on synthetic data only")

    # Create data module
    noise_config = {
        'noise_type': args.noise_type,
        'snr_range': (args.snr_min, args.snr_max)
    }

    # Use MixedDataModule if real data enabled, otherwise use synthetic only
    if args.use_real and args.real_data_dir:
        from data import MixedDataModule
        data_module = MixedDataModule(
            dataset_name=args.dataset,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            noise_config=noise_config,
            use_real=True,
            real_ratio=args.real_ratio,
            real_data_dir=args.real_data_dir
        )
    else:
        from data import AdaptiveDenoisingDataModule
        data_module = AdaptiveDenoisingDataModule(
            dataset_name=args.dataset,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            noise_config=noise_config
        )
    
    data_module.setup()

    # Create model
    model = AdaptiveTemporalDenoisingAE(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        num_scales=args.num_scales,
        num_encoder_blocks=args.num_encoder_blocks
    )

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Create trainer
    trainer = AdaptiveDenoisingTrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        optimizer=optimizer,
        scheduler=scheduler,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )

    # Train
    trainer.train(
        num_epochs=args.epochs,
        resume_from=args.resume
    )

    print("\n🎉 Training finished! Check results in:", args.log_dir)
    print("📊 Run: tensorboard --logdir", args.log_dir, "\n")


if __name__ == "__main__":
    main()
