"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  COMPREHENSIVE EVALUATION SUITE                                              ║
║  "The truth lies in the edge cases"                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Evaluates:
1. SNR improvement vs input SNR (memorization detection)
2. Spectral alignment (identity preservation)
3. Edge case performance (0-5 dB regime)
4. Cross-noise consistency
5. Synthetic-to-real generalization gap
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import signal as scipy_signal
import json

from model import AdaptiveTemporalDenoisingAE
from data import NoiseInjector, ECGDataset, AudioDataset, compute_snr_batch


# ──────────────────────────────────────────────────────────────────────────────
#  EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_spectral_distance(clean: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Compute spectral distance — measures if signal identity is preserved.
    Low distance = good identity preservation.
    High distance = model is distorting signal characteristics.
    """
    clean_fft = torch.fft.rfft(clean, dim=-1)
    recon_fft = torch.fft.rfft(reconstructed, dim=-1)
    
    # Normalized spectral distance
    clean_mag = clean_fft.abs()
    recon_mag = recon_fft.abs()
    
    # Cosine similarity in spectral domain
    similarity = torch.nn.functional.cosine_similarity(
        clean_mag.view(clean_mag.shape[0], -1),
        recon_mag.view(recon_mag.shape[0], -1),
        dim=-1
    )
    
    return similarity  # 1.0 = identical spectra


def compute_smoothness_metric(signal: torch.Tensor) -> torch.Tensor:
    """
    Compute smoothness (second derivative).
    If reconstructed is MUCH smoother than clean → over-smoothing detected.
    """
    # Second derivative as smoothness proxy
    diff2 = torch.diff(signal, n=2, dim=-1)
    return torch.mean(diff2 ** 2, dim=-1)


def compute_edge_preservation(clean: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Detect if sharp features (QRS complexes, transients) are preserved.
    """
    # Find peaks in clean signal (edges/transients)
    edge_scores = []
    
    for c, r in zip(clean, reconstructed):
        c_np = c[0].cpu().numpy()
        r_np = r[0].cpu().numpy()
        
        # Find peak locations in clean
        peaks, _ = scipy_signal.find_peaks(np.abs(c_np), height=np.std(c_np) * 0.5)
        
        if len(peaks) == 0:
            edge_scores.append(1.0)
            continue
        
        # Check if peaks are preserved in reconstructed
        peak_preservation = []
        for p in peaks:
            if p < len(r_np):
                # Compare local neighborhood
                window = 5
                start = max(0, p - window)
                end = min(len(r_np), p + window)
                
                clean_local = c_np[start:end]
                recon_local = r_np[start:end]
                
                corr = np.corrcoef(clean_local, recon_local)[0, 1]
                peak_preservation.append(corr if not np.isnan(corr) else 0)
        
        edge_scores.append(np.mean(peak_preservation) if peak_preservation else 0)
    
    return torch.tensor(edge_scores)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN EVALUATOR
# ──────────────────────────────────────────────────────────────────────────────

class ComprehensiveEvaluator:
    """
    Evaluates denoising model across all critical dimensions.
    """
    
    def __init__(
        self,
        model: AdaptiveTemporalDenoisingAE,
        device: str = 'cpu',
        results_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.results_dir = results_dir or Path(__file__).parent / 'results' / 'evaluation'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'snr_improvement': [],
            'spectral_alignment': [],
            'edge_preservation': [],
            'smoothness_ratio': [],
            'noise_type_consistency': [],
            'input_snr_bins': []
        }
    
    @torch.no_grad()
    def evaluate_snr_vs_input(
        self,
        dataset: torch.utils.data.Dataset,
        num_samples: int = 500,
        snr_bins: List[Tuple[float, float]] = None
    ) -> Dict:
        """
        Evaluate SNR improvement across different input SNR ranges.
        
        Key insight: If model only works at high SNR, it's memorizing.
        Real denoising should work even at 0-5 dB.
        """
        if snr_bins is None:
            snr_bins = [
                (0, 5),    # Extreme noise - truth lives here
                (5, 10),   # Heavy noise
                (10, 15),  # Moderate noise
                (15, 20),  # Light noise
                (20, 30),  # Very light noise
            ]
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        bin_results = {f"{low}-{high}dB": {'input_snrs': [], 'output_snrs': [], 'improvements': []}
                       for low, high in snr_bins}
        
        for noisy, clean, label_snrs in loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            output = self.model(noisy)
            reconstructed = output['output']
            
            # Compute input and output SNRs
            input_snrs = compute_snr_batch(clean, noisy)
            output_snrs = compute_snr_batch(clean, reconstructed)
            improvements = output_snrs - input_snrs
            
            # Bin by input SNR
            for i, (in_snr, out_snr, imp, label) in enumerate(zip(
                input_snrs, output_snrs, improvements, label_snrs
            )):
                in_snr_val = in_snr.item()
                
                for low, high in snr_bins:
                    if low <= in_snr_val < high:
                        bin_key = f"{low}-{high}dB"
                        bin_results[bin_key]['input_snrs'].append(in_snr_val)
                        bin_results[bin_key]['output_snrs'].append(out_snr.item())
                        bin_results[bin_key]['improvements'].append(imp.item())
                        break
        
        # Aggregate
        summary = {}
        for bin_key, data in bin_results.items():
            if data['improvements']:
                summary[bin_key] = {
                    'mean_input_snr': np.mean(data['input_snrs']),
                    'mean_output_snr': np.mean(data['output_snrs']),
                    'mean_improvement': np.mean(data['improvements']),
                    'std_improvement': np.std(data['improvements']),
                    'num_samples': len(data['improvements'])
                }
        
        self.results['snr_improvement'] = summary
        return summary
    
    @torch.no_grad()
    def evaluate_spectral_alignment(
        self,
        dataset: torch.utils.data.Dataset,
        num_samples: int = 200
    ) -> Dict:
        """
        Check if model preserves signal identity or just smooths everything.
        """
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        spectral_similarities = []
        smoothness_ratios = []
        edge_scores = []
        
        for noisy, clean, _ in loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            output = self.model(noisy)
            reconstructed = output['output']
            
            # Spectral similarity
            spec_sim = compute_spectral_distance(clean, reconstructed)
            spectral_similarities.extend(spec_sim.cpu().tolist())
            
            # Smoothness ratio (reconstructed / clean)
            clean_smooth = compute_smoothness_metric(clean)
            recon_smooth = compute_smoothness_metric(reconstructed)
            smooth_ratio = (recon_smooth / (clean_smooth + 1e-8)).cpu()
            smoothness_ratios.extend(smooth_ratio.tolist())
            
            # Edge preservation
            edge_pres = compute_edge_preservation(clean, reconstructed)
            edge_scores.extend(edge_pres.tolist())
            
            if len(spectral_similarities) >= num_samples:
                break
        
        summary = {
            'mean_spectral_similarity': np.mean(spectral_similarities),
            'std_spectral_similarity': np.std(spectral_similarities),
            'mean_smoothness_ratio': np.mean(smoothness_ratios),
            'std_smoothness_ratio': np.std(smoothness_ratios),
            'mean_edge_preservation': np.mean(edge_scores),
            'std_edge_preservation': np.std(edge_scores),
            'num_samples': len(spectral_similarities)
        }
        
        # Interpretation
        if summary['mean_spectral_similarity'] < 0.85:
            summary['warning'] = "LOW spectral similarity — model may be distorting signal identity"
        if summary['mean_smoothness_ratio'] < 0.5:
            summary['warning'] = "OVER-SMOOTHING detected — model is blurring sharp features"
        if summary['mean_edge_preservation'] < 0.7:
            summary['warning'] = "POOR edge preservation — transients/peaks being lost"
        
        self.results['spectral_alignment'] = summary
        return summary
    
    @torch.no_grad()
    def evaluate_noise_type_consistency(
        self,
        base_dataset: torch.utils.data.Dataset,
        noise_types: List[str] = ['gaussian', 'poisson', 'impulse', 'colored'],
        num_samples: int = 100
    ) -> Dict:
        """
        Test if model works across different noise types.
        If performance varies wildly → not robust.
        """
        noise_results = {}
        
        for noise_type in noise_types:
            # Create noise injector for this type
            injector = NoiseInjector(noise_type=noise_type, snr_range=(5, 20))
            
            input_snrs = []
            output_snrs = []
            improvements = []
            
            for i in range(min(num_samples, len(base_dataset))):
                _, clean, _ = base_dataset[i]
                clean_np = clean[0].numpy()
                
                # Add this noise type
                noisy_np = injector(clean_np)
                
                noisy = torch.FloatTensor(noisy_np).unsqueeze(0).to(self.device)
                clean_t = torch.FloatTensor(clean_np).unsqueeze(0).to(self.device)
                
                output = self.model(noisy)
                reconstructed = output['output']
                
                in_snr = compute_snr_batch(clean_t, noisy).item()
                out_snr = compute_snr_batch(clean_t, reconstructed).item()
                
                input_snrs.append(in_snr)
                output_snrs.append(out_snr)
                improvements.append(out_snr - in_snr)
            
            noise_results[noise_type] = {
                'mean_input_snr': np.mean(input_snrs),
                'mean_output_snr': np.mean(output_snrs),
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements)
            }
        
        # Check consistency
        improvement_stds = [r['std_improvement'] for r in noise_results.values()]
        improvement_means = [r['mean_improvement'] for r in noise_results.values()]
        
        consistency_summary = {
            'noise_types': noise_results,
            'improvement_variance_across_types': np.var(improvement_means),
            'warning': None
        }
        
        if consistency_summary['improvement_variance_across_types'] > 2.0:
            consistency_summary['warning'] = "HIGH variance across noise types — model not robust"
        
        self.results['noise_type_consistency'] = consistency_summary
        return consistency_summary
    
    @torch.no_grad()
    def evaluate_memorization_vs_learning(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        num_samples: int = 200
    ) -> Dict:
        """
        Detect if model is memorizing training distribution vs actually learning to denoise.
        
        Key test: Performance gap between train and test distributions.
        """
        def evaluate_on_dataset(dataset):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                num_workers=0
            )
            
            improvements = []
            for noisy, clean, _ in loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                output = self.model(noisy)
                reconstructed = output['output']
                
                in_snr = compute_snr_batch(clean, noisy)
                out_snr = compute_snr_batch(clean, reconstructed)
                improvements.extend((out_snr - in_snr).cpu().tolist())
                
                if len(improvements) >= num_samples:
                    break
            
            return improvements
        
        train_improvements = evaluate_on_dataset(train_dataset)
        test_improvements = evaluate_on_dataset(test_dataset)
        
        gap = np.mean(train_improvements) - np.mean(test_improvements)
        
        summary = {
            'train_mean_improvement': np.mean(train_improvements),
            'test_mean_improvement': np.mean(test_improvements),
            'train_test_gap': gap,
            'warning': None
        }
        
        if gap > 3.0:  # More than 3 dB gap
            summary['warning'] = f"MEMORIZATION detected! Train-Test gap: {gap:.2f} dB"
        elif gap > 1.0:
            summary['warning'] = f"Moderate generalization gap: {gap:.2f} dB (acceptable)"
        
        self.results['memorization_detection'] = summary
        return summary
    
    def generate_report(self) -> str:
        """Generate human-readable evaluation report."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("COMPREHENSIVE EVALUATION REPORT")
        report.append("=" * 80)
        
        # SNR Improvement
        report.append("\n📊 SNR IMPROVEMENT VS INPUT SNR")
        report.append("-" * 60)
        for bin_name, data in self.results['snr_improvement'].items():
            if data.get('num_samples', 0) > 0:
                report.append(f"  {bin_name}:")
                report.append(f"    Input:  {data['mean_input_snr']:.2f} dB")
                report.append(f"    Output: {data['mean_output_snr']:.2f} dB")
                report.append(f"    Improvement: {data['mean_improvement']:.2f} ± {data['std_improvement']:.2f} dB")
                report.append(f"    Samples: {data['num_samples']}")
        
        # Spectral Alignment
        report.append("\n🎯 SPECTRAL ALIGNMENT")
        report.append("-" * 60)
        spec = self.results['spectral_alignment']
        if spec:
            report.append(f"  Spectral Similarity: {spec['mean_spectral_similarity']:.3f} ± {spec['std_spectral_similarity']:.3f}")
            report.append(f"  Smoothness Ratio: {spec['mean_smoothness_ratio']:.3f} ± {spec['std_smoothness_ratio']:.3f}")
            report.append(f"  Edge Preservation: {spec['mean_edge_preservation']:.3f} ± {spec['std_edge_preservation']:.3f}")
            if 'warning' in spec and spec['warning']:
                report.append(f"  ⚠️  {spec['warning']}")
        
        # Noise Type Consistency
        report.append("\n🔀 NOISE TYPE CONSISTENCY")
        report.append("-" * 60)
        noise = self.results['noise_type_consistency']
        if noise:
            for nt, data in noise['noise_types'].items():
                report.append(f"  {nt}: {data['mean_improvement']:.2f} ± {data['std_improvement']:.2f} dB")
            report.append(f"  Variance across types: {noise['improvement_variance_across_types']:.2f}")
            if noise.get('warning'):
                report.append(f"  ⚠️  {noise['warning']}")
        
        # Memorization Detection
        report.append("\n🧠 MEMORIZATION VS LEARNING")
        report.append("-" * 60)
        mem = self.results['memorization_detection']
        if mem:
            report.append(f"  Train Improvement: {mem['train_mean_improvement']:.2f} dB")
            report.append(f"  Test Improvement: {mem['test_mean_improvement']:.2f} dB")
            report.append(f"  Gap: {mem['train_test_gap']:.2f} dB")
            if mem.get('warning'):
                report.append(f"  ⚠️  {mem['warning']}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "evaluation_report.txt"):
        """Save report to file."""
        report = self.generate_report()
        with open(self.results_dir / filename, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {self.results_dir / filename}")
        
        # Also save raw data
        with open(self.results_dir / 'evaluation_data.json', 'w') as f:
            # Convert numpy types for JSON
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            json.dump(convert(self.results), f, indent=2)
        print(f"📊 Raw data saved to: {self.results_dir / 'evaluation_data.json'}")


# ──────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_snr_improvement_curve(results: Dict, save_path: Optional[Path] = None):
    """Plot SNR improvement vs input SNR."""
    snr_data = results['snr_improvement']
    
    if not snr_data:
        print("No SNR data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bin_centers = []
    improvements = []
    errors = []
    
    for bin_name, data in snr_data.items():
        if data.get('num_samples', 0) > 0:
            # Extract bin center
            low, high = map(float, bin_name.replace('dB', '').split('-'))
            bin_centers.append((low + high) / 2)
            improvements.append(data['mean_improvement'])
            errors.append(data['std_improvement'])
    
    ax.errorbar(bin_centers, improvements, yerr=errors, fmt='o-', capsize=5,
                color='#00ff88', markersize=8, linewidth=2)
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Input SNR (dB)', fontsize=12)
    ax.set_ylabel('SNR Improvement (dB)', fontsize=12)
    ax.set_title('Denoising Performance vs Input Noise Level', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Highlight critical region
    ax.axvspan(0, 5, alpha=0.2, color='red', label='Critical regime (0-5 dB)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == '__main__':
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "COMPREHENSIVE EVALUATION SUITE" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # Load model
    checkpoint_path = Path(__file__).parent / 'results' / 'checkpoints' / 'best.pt'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Train the model first!")
        exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = AdaptiveTemporalDenoisingAE(
        in_channels=1,
        hidden_channels=64,
        num_scales=3,
        num_encoder_blocks=4
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Loaded model from: {checkpoint_path}")
    print(f"🔥 Device: {device.upper()}\n")
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(model, device=device)
    
    # Create test datasets
    print("📂 Creating evaluation datasets...")
    
    test_dataset = ECGDataset(
        seq_len=1024,
        num_samples=500,
        noise_config={'noise_type': 'mixed', 'snr_range': (0, 30)},
        seed=9999
    )
    
    # Run evaluations
    print("\n🔬 Running evaluations...\n")
    
    print("1. SNR Improvement vs Input SNR...")
    evaluator.evaluate_snr_vs_input(test_dataset, num_samples=500)
    
    print("2. Spectral Alignment...")
    evaluator.evaluate_spectral_alignment(test_dataset, num_samples=200)
    
    print("3. Noise Type Consistency...")
    evaluator.evaluate_noise_type_consistency(test_dataset, num_samples=100)
    
    print("4. Memorization Detection...")
    train_dataset = ECGDataset(
        seq_len=1024,
        num_samples=500,
        noise_config={'noise_type': 'mixed', 'snr_range': (5, 20)},
        seed=42
    )
    evaluator.evaluate_memorization_vs_learning(train_dataset, test_dataset, num_samples=200)
    
    # Generate report
    print("\n" + "=" * 80)
    report = evaluator.generate_report()
    print(report)
    
    # Save
    evaluator.save_report()
    
    # Plot
    plot_snr_improvement_curve(
        evaluator.results,
        save_path=evaluator.results_dir / 'snr_improvement_curve.png'
    )
    
    print("\n✅ Evaluation complete!\n")
