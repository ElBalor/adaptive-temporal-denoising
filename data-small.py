"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DATA PIPELINES & NOISE INJECTION                                            ║
║  "Corrupting perfection since 2026"                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import scipy.io as sio
from scipy import signal as scipy_signal
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
#  NOISE INJECTION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class NoiseInjector:
    """
    Multi-modal noise injection for training robust denoising models.
    Supports various noise types with controllable intensity.
    """
    
    NOISE_TYPES = ['gaussian', 'poisson', 'impulse', 'colored', 'mixed']
    
    def __init__(
        self,
        noise_type: str = 'mixed',
        snr_range: Tuple[float, float] = (5.0, 30.0),
        impulse_prob: float = 0.05,
        seed: Optional[int] = None
    ):
        self.noise_type = noise_type
        self.snr_min, self.snr_max = snr_range
        self.impulse_prob = impulse_prob
        self.rng = np.random.default_rng(seed)
        
    def _compute_noise_power(self, signal_power: float, snr_db: float) -> float:
        """Calculate noise power for target SNR."""
        return signal_power / (10 ** (snr_db / 10))
    
    def add_gaussian_noise(self, x: np.ndarray, snr_db: float) -> np.ndarray:
        """Add white Gaussian noise."""
        signal_power = np.mean(x ** 2)
        noise_power = self._compute_noise_power(signal_power, snr_db)
        noise = self.rng.normal(0, np.sqrt(noise_power), x.shape)
        return x + noise.astype(np.float32)
    
    def add_poisson_noise(self, x: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Poisson noise (shot noise)."""
        # Shift to positive range
        x_shifted = x - x.min() + 1e-6
        x_scaled = x_shifted / x_shifted.max() * 1000
        
        # Poisson noise
        noisy = self.rng.poisson(x_scaled)
        noisy = (noisy - noisy.mean()) / (noisy.std() + 1e-8)
        
        # Scale to match SNR
        signal_power = np.mean(x ** 2)
        noise_power = np.mean((noisy - x) ** 2)
        scale = np.sqrt(signal_power / (noise_power + 1e-8) / (10 ** (snr_db / 10)))
        
        return x + (noisy - x) * scale
    
    def add_impulse_noise(self, x: np.ndarray, snr_db: float) -> np.ndarray:
        """Add impulse/salt-pepper noise."""
        noisy = x.copy()
        mask = self.rng.random(x.shape) < self.impulse_prob
        
        # Random impulses (positive or negative)
        impulses = self.rng.choice([-1, 1], size=x.shape) * self.rng.uniform(0.5, 1.0, x.shape)
        noisy[mask] = x[mask] + impulses[mask] * np.std(x)
        
        # Scale to match SNR
        signal_power = np.mean(x ** 2)
        noise_power = np.mean((noisy - x) ** 2)
        if noise_power > 0:
            scale = np.sqrt(signal_power / noise_power / (10 ** (snr_db / 10)))
            noisy = x + (noisy - x) * scale
        
        return noisy
    
    def add_colored_noise(self, x: np.ndarray, snr_db: float, beta: float = 1.0) -> np.ndarray:
        """
        Add colored noise (1/f^beta).
        beta=0: white, beta=1: pink, beta=2: brown
        """
        n = len(x)
        
        # Generate white noise
        white = self.rng.normal(0, 1, n)
        
        # Color the noise via FFT
        fft = np.fft.fft(white)
        freqs = np.fft.fftfreq(n)
        
        # Apply coloration filter
        with np.errstate(divide='ignore', invalid='ignore'):
            coloration = 1.0 / (np.abs(freqs) + 1e-6) ** (beta / 2)
        coloration[0] = 0  # DC component
        
        colored = np.fft.ifft(fft * coloration).real
        
        # Scale to target SNR
        signal_power = np.mean(x ** 2)
        noise_power = np.mean(colored ** 2)
        colored = colored * np.sqrt(signal_power / noise_power / (10 ** (snr_db / 10)))
        
        return x + colored.astype(np.float32)
    
    def __call__(self, x: np.ndarray, snr_db: Optional[float] = None) -> np.ndarray:
        """
        Apply noise to input signal.
        
        Args:
            x: Clean signal [T] or [B, T]
            snr_db: Optional fixed SNR (random if None)
            
        Returns:
            Noisy signal
        """
        # Handle batch dimension
        if x.ndim == 1:
            x = x[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        
        if snr_db is None:
            snr_db = self.rng.uniform(self.snr_min, self.snr_max)
        
        noisy_batch = []
        
        for sig in x:
            if self.noise_type == 'gaussian':
                noisy = self.add_gaussian_noise(sig, snr_db)
            elif self.noise_type == 'poisson':
                noisy = self.add_poisson_noise(sig, snr_db)
            elif self.noise_type == 'impulse':
                noisy = self.add_impulse_noise(sig, snr_db)
            elif self.noise_type == 'colored':
                beta = self.rng.uniform(0.5, 1.5)  # Vary coloration
                noisy = self.add_colored_noise(sig, snr_db, beta)
            elif self.noise_type == 'mixed':
                # Random combination
                choice = self.rng.choice(['gaussian', 'poisson', 'impulse', 'colored'])
                if choice == 'colored':
                    beta = self.rng.uniform(0.5, 1.5)
                    noisy = self.add_colored_noise(sig, snr_db, beta)
                elif choice == 'gaussian':
                    noisy = self.add_gaussian_noise(sig, snr_db)
                elif choice == 'poisson':
                    noisy = self.add_poisson_noise(sig, snr_db)
                else:
                    noisy = self.add_impulse_noise(sig, snr_db)
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")
            
            noisy_batch.append(noisy)
        
        result = np.stack(noisy_batch)
        return result[0] if squeeze else result


# ──────────────────────────────────────────────────────────────────────────────
#  DATASETS
# ──────────────────────────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    """
    ECG signal denoising dataset.
    Uses PhysioNet PTB-XL or simulated signals.
    """
    
    def __init__(
        self,
        seq_len: int = 1024,
        num_samples: int = 10000,
        noise_config: Optional[Dict] = None,
        data_dir: Optional[str] = None,
        split: str = 'train',
        seed: int = 42
    ):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.noise_injector = NoiseInjector(**(noise_config or {}))
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / 'data'
        self.split = split
        self.rng = np.random.default_rng(seed)
        
        # Generate or load ECG signals
        self.signals = self._generate_ecg_signals()
        
    def _generate_ecg_signals(self) -> np.ndarray:
        """Generate synthetic ECG signals with realistic morphology."""
        signals = []
        
        for _ in range(self.num_samples):
            # Random heart rate (60-120 bpm)
            hr = self.rng.uniform(60, 120)
            
            # Generate ECG using modified Gaussian model
            t = np.linspace(0, 2, self.seq_len)
            
            # P wave
            p_amp = self.rng.uniform(0.1, 0.3)
            p_center = 0.2 + self.rng.uniform(-0.05, 0.05)
            p_width = 0.04 + self.rng.uniform(0, 0.02)
            p_wave = p_amp * np.exp(-((t - p_center) ** 2) / (2 * p_width ** 2))
            
            # QRS complex
            q_amp = self.rng.uniform(-0.1, -0.05)
            r_amp = self.rng.uniform(0.8, 1.2)
            s_amp = self.rng.uniform(-0.2, -0.1)
            
            q_wave = q_amp * np.exp(-((t - 0.35) ** 2) / (2 * 0.01 ** 2))
            r_wave = r_amp * np.exp(-((t - 0.4) ** 2) / (2 * 0.015 ** 2))
            s_wave = s_amp * np.exp(-((t - 0.45) ** 2) / (2 * 0.01 ** 2))
            qrs = q_wave + r_wave + s_wave
            
            # T wave
            t_amp = self.rng.uniform(0.2, 0.4)
            t_center = 0.6 + self.rng.uniform(-0.05, 0.05)
            t_width = 0.1 + self.rng.uniform(0, 0.05)
            t_wave = t_amp * np.exp(-((t - t_center) ** 2) / (2 * t_width ** 2))
            
            # U wave (sometimes)
            if self.rng.random() > 0.5:
                u_amp = self.rng.uniform(0.05, 0.1)
                u_wave = u_amp * np.exp(-((t - 0.85) ** 2) / (2 * 0.03 ** 2))
            else:
                u_wave = 0
            
            # Combine
            ecg = p_wave + qrs + t_wave + u_wave
            
            # Add baseline wander
            baseline = 0.05 * np.sin(2 * np.pi * 0.5 * t + self.rng.uniform(0, 2 * np.pi))
            ecg += baseline
            
            # Normalize
            ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
            
            signals.append(ecg)
        
        return np.array(signals, dtype=np.float32)[:, np.newaxis, :]
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        clean = self.signals[idx].copy()
        noisy = self.noise_injector(clean[0])[np.newaxis, :]
        
        snr = 10 * np.log10(
            np.mean(clean ** 2) / (np.mean((clean - noisy) ** 2) + 1e-8)
        )
        
        return (
            torch.FloatTensor(noisy),
            torch.FloatTensor(clean),
            snr
        )


class AudioDataset(Dataset):
    """
    Audio denoising dataset.
    Uses speech signals with various noise corruptions.
    """
    
    def __init__(
        self,
        seq_len: int = 16384,
        num_samples: int = 5000,
        noise_config: Optional[Dict] = None,
        sample_rate: int = 16000,
        seed: int = 42
    ):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.noise_injector = NoiseInjector(**(noise_config or {'noise_type': 'mixed', 'snr_range': (0, 20)}))
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)
        
        self.signals = self._generate_speech_signals()
    
    def _generate_speech_signals(self) -> np.ndarray:
        """Generate speech-like signals using formant synthesis."""
        signals = []
        
        for _ in range(self.num_samples):
            t = np.linspace(0, self.seq_len / self.sample_rate, self.seq_len)
            
            # Random fundamental frequency (male/female)
            f0 = self.rng.choice([120, 220]) + self.rng.uniform(-20, 20)
            
            # Generate voiced source (harmonic series)
            source = np.zeros_like(t)
            for h in range(1, 15):
                amp = 1.0 / h
                phase = self.rng.uniform(0, 2 * np.pi)
                source += amp * np.sin(2 * np.pi * h * f0 * t + phase)
            
            # Apply formant filter (vowel-like)
            formants = [
                (self.rng.uniform(300, 800), self.rng.uniform(50, 100)),
                (self.rng.uniform(900, 2000), self.rng.uniform(100, 200)),
                (self.rng.uniform(2500, 3500), self.rng.uniform(150, 300)),
            ]
            
            for fc, bw in formants:
                b, a = scipy_signal.butter(2, [fc - bw/2, fc + bw/2], 
                                           btype='band', fs=self.sample_rate)
                source = scipy_signal.filtfilt(b, a, source)
            
            # Amplitude envelope (syllable-like)
            envelope = np.ones_like(t)
            num_syllables = self.rng.integers(2, 5)
            for _ in range(num_syllables):
                center = self.rng.uniform(0.2, 0.8)
                width = self.rng.uniform(0.1, 0.3)
                envelope *= 0.5 + 0.5 * np.exp(-((t - center * t.max()) ** 2) / (2 * (width * t.max()) ** 2))
            
            speech = source * envelope
            speech = (speech - speech.mean()) / (speech.std() + 1e-8)
            
            signals.append(speech.astype(np.float32))
        
        return np.array(signals, dtype=np.float32)[:, np.newaxis, :]
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        clean = self.signals[idx].copy()
        noisy = self.noise_injector(clean[0])[np.newaxis, :]
        
        snr = 10 * np.log10(
            np.mean(clean ** 2) / (np.mean((clean - noisy) ** 2) + 1e-8)
        )
        
        return (
            torch.FloatTensor(noisy),
            torch.FloatTensor(clean),
            snr
        )


class SensorDataset(Dataset):
    """
    Sensor signal denoising (accelerometer, gyroscope, etc.).
    Based on UCI HAR dataset patterns.
    """
    
    def __init__(
        self,
        seq_len: int = 512,
        num_samples: int = 8000,
        noise_config: Optional[Dict] = None,
        seed: int = 42
    ):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.noise_injector = NoiseInjector(**(noise_config or {'noise_type': 'mixed', 'snr_range': (5, 25)}))
        self.rng = np.random.default_rng(seed)
        
        self.signals = self._generate_sensor_signals()
    
    def _generate_sensor_signals(self) -> np.ndarray:
        """Generate human activity sensor signals."""
        signals = []
        
        activities = ['walking', 'running', 'stairs', 'sitting', 'standing']
        
        for _ in range(self.num_samples):
            t = np.linspace(0, 2.56, self.seq_len)  # 2.56s at 20Hz
            
            activity = self.rng.choice(activities)
            
            if activity == 'walking':
                freq = self.rng.uniform(1.5, 2.5)
                signal = 0.5 * np.sin(2 * np.pi * freq * t)
                signal += 0.2 * np.sin(2 * np.pi * 2 * freq * t)
                
            elif activity == 'running':
                freq = self.rng.uniform(2.5, 4.0)
                signal = 0.8 * np.sin(2 * np.pi * freq * t)
                signal += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
                
            elif activity == 'stairs':
                freq = self.rng.uniform(1.0, 2.0)
                signal = 0.4 * np.sin(2 * np.pi * freq * t)
                signal += 0.3 * np.random.randn(self.seq_len) * 0.1
                
            else:  # sitting, standing
                signal = 0.1 * np.random.randn(self.seq_len)
                signal += 0.05 * np.sin(2 * np.pi * 0.5 * t)
            
            # Add gravity component
            signal += self.rng.uniform(-1, 1)
            
            # Normalize
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)
            
            signals.append(signal.astype(np.float32))
        
        return np.array(signals, dtype=np.float32)[:, np.newaxis, :]
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        clean = self.signals[idx].copy()
        noisy = self.noise_injector(clean[0])[np.newaxis, :]
        
        snr = 10 * np.log10(
            np.mean(clean ** 2) / (np.mean((clean - noisy) ** 2) + 1e-8)
        )
        
        return (
            torch.FloatTensor(noisy),
            torch.FloatTensor(clean),
            snr
        )


# ──────────────────────────────────────────────────────────────────────────────
#  DATA MODULE
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveDenoisingDataModule:
    """
    Complete data module with train/val/test splits.
    Supports multiple datasets and automatic downloading.
    """
    
    DATASETS = {
        'ecg': ECGDataset,
        'audio': AudioDataset,
        'sensor': SensorDataset
    }
    
    def __init__(
        self,
        dataset_name: str = 'ecg',
        seq_len: int = 1024,
        batch_size: int = 64,
        num_workers: int = 4,
        noise_config: Optional[Dict] = None,
        data_dir: Optional[str] = None
    ):
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_config = noise_config or {}
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / 'data'
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, train_size: int = 8000, val_size: int = 1000, test_size: int = 1000):
        """Initialize datasets."""
        dataset_cls = self.DATASETS.get(self.dataset_name)
        
        if dataset_cls is None:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        self.train_dataset = dataset_cls(
            seq_len=self.seq_len,
            num_samples=train_size,
            noise_config=self.noise_config
        )
        
        self.val_dataset = dataset_cls(
            seq_len=self.seq_len,
            num_samples=val_size,
            noise_config=self.noise_config,
            seed=1234
        )
        
        self.test_dataset = dataset_cls(
            seq_len=self.seq_len,
            num_samples=test_size,
            noise_config=self.noise_config,
            seed=5678
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# ──────────────────────────────────────────────────────────────────────────────
#  METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Compute output SNR in dB (single sample)."""
    noise = clean - denoised
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-8))


def compute_snr_batch(clean: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample SNR, then average.
    
    Args:
        clean: Ground truth signal [B, C, T]
        denoised: Denoised signal [B, C, T]
        
    Returns:
        Mean SNR across batch (scalar tensor)
    """
    snrs = []
    for c, d in zip(clean, denoised):
        signal_power = torch.mean(c ** 2)
        noise_power = torch.mean((c - d) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        snrs.append(snr)
    return torch.stack(snrs).mean()


def compute_mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((clean - denoised) ** 2)


def compute_sdr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Compute signal-to-distortion ratio."""
    noise = clean - denoised
    sdr = np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-8)
    return 10 * np.log10(sdr)


def compute_sdr_batch(clean: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample SDR, then average.
    
    Args:
        clean: Ground truth signal [B, C, T]
        denoised: Denoised signal [B, C, T]
        
    Returns:
        Mean SDR across batch (scalar tensor)
    """
    sdrs = []
    for c, d in zip(clean, denoised):
        signal_power = torch.sum(c ** 2)
        noise_power = torch.sum((c - d) ** 2)
        sdr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        sdrs.append(sdr)
    return torch.stack(sdrs).mean()


# ──────────────────────────────────────────────────────────────────────────────
#  REAL DATASETS (The Good Shit)
# ──────────────────────────────────────────────────────────────────────────────

class PhysioNetECG(Dataset):
    """
    Real ECG from PhysioNet PTB-XL.
    21,837 patient records. Clinical-grade data.
    
    Download: https://physionet.org/content/ptb-xl/
    """
    
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 1024,
        split: str = 'train',
        noise_config: Optional[Dict] = None,
        lead: int = 0,  # Which lead to use (0-11 for 12-lead ECG)
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.split = split
        self.lead = lead
        self.noise_injector = NoiseInjector(**(noise_config or {}))
        self.rng = np.random.default_rng(seed)
        
        # Try to load WFDB files
        self.records = self._load_records()
        
    def _load_records(self) -> List[np.ndarray]:
        """Load ECG records from WFDB format or Cache."""
        try:
            import wfdb
            import pandas as pd
            from scipy.signal import resample

            # Cache path (Split-specific to prevent leakage)
            cache_path = self.data_dir / f'ptb-xl-cache-{self.split}.pt'

            # Check cache
            if cache_path.exists():
                print(f"⚡ Loading PTB-XL {self.split} from cache: {cache_path}")
                records = torch.load(cache_path, weights_only=False)
                print(f"✅ Loaded {len(records)} records from cache (Instant!)")
                return records

            print(f"📥 Loading PhysioNet PTB-XL from {self.data_dir}...")
            print(f"   (First run: building cache for future speed...)")

            # Find the CSV database file
            csv_file = self.data_dir / 'ptbxl_database.csv'
            if not csv_file.exists():
                # Try alternative locations
                for alt in ['ptb-xl.csv', 'database.csv']:
                    alt_file = self.data_dir / alt
                    if alt_file.exists():
                        csv_file = alt_file
                        break

            if not csv_file.exists():
                print(f"⚠️  PTB-XL database CSV not found at {csv_file}")
                print(f"   Expected: ptbxl_database.csv")
                return []

            # Parse CSV
            df = pd.read_csv(csv_file)

            # Filter by split (strat_fold column)
            if 'strat_fold' in df.columns:
                if self.split == 'train':
                    df = df[df['strat_fold'] != 10]  # Fold 10 is test
                else:
                    df = df[df['strat_fold'] == 10]
            else:
                # No fold column - use all data for train, empty for val
                if self.split == 'val':
                    print(f"⚠️  No strat_fold column - using 20% of data for validation")
                    n_val = int(len(df) * 0.2)
                    df = df.iloc[-n_val:]
                else:
                    df = df.iloc[:-int(len(df) * 0.2)]

            # Find records directory (records100 or records500)
            records_dir = None
            for d in ['records100', 'records500']:
                if (self.data_dir / d).exists():
                    records_dir = self.data_dir / d
                    break

            if records_dir is None:
                print(f"⚠️  No records100/ or records500/ found in {self.data_dir}")
                return []

            print(f"📂 Using records from: {records_dir}")
            print(f"📊 Loading {len(df)} records for {self.split} split...")
            print(f"   📋 CSV Columns: {list(df.columns)[:5]}...") # Debug: show columns

            records = []
            loaded = 0
            failed = 0

            for idx, row in df.iterrows():
                # Get filename (try different column names)
                filename = None
                for col in ['filename_lr', 'filename_hr', 'ecg_id', 'id']:
                    if col in df.columns and pd.notna(row.get(col)):
                        filename = str(row[col])
                        break

                if filename is None:
                    failed += 1
                    if failed == 1:
                        print(f"   ⚠️  Filename not found in columns. Available: {list(df.columns)}")
                    continue

                # Remove extension if present
                filename = filename.replace('.dat', '').replace('.hea', '')
                
                # Strip directory prefix if present (e.g., "records100/00000/...")
                # Because we are already searching INSIDE records100/
                for prefix in ['records100/', 'records500/']:
                    if filename.startswith(prefix):
                        filename = filename[len(prefix):]
                        break

                # Find the file in nested directories
                record_path = None
                # Try exact match first (e.g. 00000/00001_lr)
                target = records_dir / filename
                if target.with_suffix('.dat').exists():
                    record_path = target
                elif target.with_suffix('.hea').exists():
                    record_path = target
                else:
                    # Fallback to rglob if structure is weird
                    matches = list(records_dir.rglob(f'{os.path.basename(filename)}*'))
                    if matches:
                        record_path = matches[0].with_suffix('')

                if record_path is None:
                    failed += 1
                    if failed == 1:
                        print(f"   ⚠️  Example: Could not find file for '{filename}' in {records_dir}")
                    continue

                try:
                    # Load WFDB record
                    record = wfdb.rdrecord(str(record_path))
                    ecg = record.p_signal[:, self.lead]  # Extract selected lead

                    # Resample to seq_len if needed
                    if len(ecg) > self.seq_len:
                        # Random crop
                        start = self.rng.integers(0, len(ecg) - self.seq_len)
                        ecg = ecg[start:start + self.seq_len]
                    elif len(ecg) < self.seq_len:
                        # Resample
                        ecg = resample(ecg, self.seq_len)

                    # Normalize
                    ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
                    records.append(ecg.astype(np.float32))
                    loaded += 1

                except Exception as e:
                    failed += 1
                    if failed <= 3:
                        print(f"   ⚠️  Error #{failed} loading '{filename}':")
                        print(f"       Path: {record_path}")
                        print(f"       Error: {type(e).__name__}: {e}")
                    continue

            # Save cache (Split-specific)
            if len(records) > 0:
                print(f"\n💾 Saving {self.split} cache to {cache_path}...")
                torch.save(records, cache_path)
                print(f"✅ {self.split.capitalize()} cache saved! Next load will be instant.")

            print(f"✅ Loaded {loaded} ECG records from PTB-XL ({failed} failed)")
            return records

        except ImportError:
            print("⚠️  wfdb not installed. Install with: pip install wfdb")
            return []
        except Exception as e:
            print(f"❌ Error loading PTB-XL: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def __len__(self) -> int:
        return max(len(self.records) * 10, 1000)  # Augment by repeating
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        # Randomly select and augment
        record_idx = idx % len(self.records)
        clean_ecg = self.records[record_idx].copy()
        
        # Random crop if longer than seq_len
        if len(clean_ecg) > self.seq_len:
            start = self.rng.integers(0, len(clean_ecg) - self.seq_len)
            clean_ecg = clean_ecg[start:start + self.seq_len]
        
        # Add noise
        noisy_ecg = self.noise_injector(clean_ecg)
        
        # Compute SNR
        snr = 10 * np.log10(
            np.mean(clean_ecg ** 2) / (np.mean((clean_ecg - noisy_ecg) ** 2) + 1e-8)
        )
        
        return (
            torch.FloatTensor(noisy_ecg).unsqueeze(0),
            torch.FloatTensor(clean_ecg).unsqueeze(0),
            snr
        )


class LibriSpeech(Dataset):
    """
    Real speech from LibriSpeech.
    1000 hours of read English speech.
    
    Download: https://www.openslr.org/12
    """
    
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 16384,
        split: str = 'train',
        noise_config: Optional[Dict] = None,
        sample_rate: int = 16000,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.split = split
        self.sample_rate = sample_rate
        self.noise_injector = NoiseInjector(**(noise_config or {'noise_type': 'mixed', 'snr_range': (0, 20)}))
        self.rng = np.random.default_rng(seed)
        
        self.audio_files = self._scan_files()
        
    def _scan_files(self) -> List[Path]:
        """Scan for FLAC files."""
        try:
            import soundfile as sf
        except ImportError:
            print("⚠️  soundfile not installed. Install with: pip install soundfile")
            return []
        
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            print(f"⚠️  LibriSpeech {self.split} not found at {split_dir}")
            return []
        
        # Find all FLAC files
        flac_files = list(split_dir.rglob('*.flac'))
        print(f"✅ Found {len(flac_files)} audio files in LibriSpeech/{self.split}")
        return flac_files
    
    def __len__(self) -> int:
        return max(len(self.audio_files) * 5, 1000)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        import soundfile as sf
        
        # Select file
        file_idx = idx % len(self.audio_files)
        audio_path = self.audio_files[file_idx]
        
        # Load audio
        audio, sr = sf.read(audio_path, dtype=np.float32)
        
        # Resample if needed
        if sr != self.sample_rate:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * self.sample_rate / sr))
        
        # Random crop
        if len(audio) > self.seq_len:
            start = self.rng.integers(0, len(audio) - self.seq_len)
            audio = audio[start:start + self.seq_len]
        elif len(audio) < self.seq_len:
            # Pad
            audio = np.pad(audio, (0, self.seq_len - len(audio)), mode='constant')
        
        # Normalize
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)
        
        # Add noise
        noisy = self.noise_injector(audio)
        
        # Compute SNR
        snr = 10 * np.log10(
            np.mean(audio ** 2) / (np.mean((audio - noisy) ** 2) + 1e-8)
        )
        
        return (
            torch.FloatTensor(noisy).unsqueeze(0),
            torch.FloatTensor(audio).unsqueeze(0),
            snr
        )


class UCIHARDataset(Dataset):
    """
    Real sensor data from UCI HAR.
    Accelerometer + gyroscope from 30 subjects.
    
    Download: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    """
    
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 512,
        split: str = 'train',
        noise_config: Optional[Dict] = None,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.split = split
        self.noise_injector = NoiseInjector(**(noise_config or {'noise_type': 'mixed', 'snr_range': (5, 25)}))
        self.rng = np.random.default_rng(seed)
        
        self.signals = self._load_data()
        
    def _load_data(self) -> np.ndarray:
        """Load UCI HAR sensor data."""
        split_folder = 'train' if self.split == 'train' else 'test'
        data_path = self.data_dir / split_folder
        
        if not data_path.exists():
            print(f"⚠️  UCI HAR not found at {data_path}")
            return np.array([], dtype=np.float32)
        
        # Load accelerometer data
        acc_path = data_path / 'Inertial Signals'
        
        try:
            # Load total acceleration (X, Y, Z)
            acc_x = np.loadtxt(acc_path / 'total_acc_x_train.txt')
            acc_y = np.loadtxt(acc_path / 'total_acc_y_train.txt')
            acc_z = np.loadtxt(acc_path / 'total_acc_z_train.txt')
            
            # Combine axes
            signals = np.stack([acc_x, acc_y, acc_z], axis=1)
            
            # Normalize per signal
            for i in range(len(signals)):
                signals[i] = (signals[i] - signals[i].mean()) / (signals[i].std() + 1e-8)
            
            print(f"✅ Loaded {len(signals)} sensor sequences from UCI HAR")
            return signals.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error loading UCI HAR: {e}")
            return np.array([], dtype=np.float32)
    
    def __len__(self) -> int:
        return max(len(self.signals) * 5, 500)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        # Select signal (use X axis for 1D denoising)
        signal_idx = idx % len(self.signals)
        clean_signal = self.signals[signal_idx, 0, :]  # X axis
        
        # Random crop
        if len(clean_signal) > self.seq_len:
            start = self.rng.integers(0, len(clean_signal) - self.seq_len)
            clean_signal = clean_signal[start:start + self.seq_len]
        elif len(clean_signal) < self.seq_len:
            clean_signal = np.pad(clean_signal, (0, self.seq_len - len(clean_signal)), mode='constant')
        
        # Add noise
        noisy_signal = self.noise_injector(clean_signal)
        
        # Compute SNR
        snr = 10 * np.log10(
            np.mean(clean_signal ** 2) / (np.mean((clean_signal - noisy_signal) ** 2) + 1e-8)
        )
        
        return (
            torch.FloatTensor(noisy_signal).unsqueeze(0),
            torch.FloatTensor(clean_signal).unsqueeze(0),
            snr
        )


# ──────────────────────────────────────────────────────────────────────────────
#  MIXED DATA MODULE (Synthetic + Real)
# ──────────────────────────────────────────────────────────────────────────────

class MixedDataModule:
    """
    Combines synthetic and real datasets for robust training.
    """
    
    def __init__(
        self,
        dataset_name: str = 'ecg',
        seq_len: int = 1024,
        batch_size: int = 64,
        num_workers: int = 4,
        noise_config: Optional[Dict] = None,
        use_real: bool = False,
        real_ratio: float = 0.3,
        real_data_dir: Optional[str] = None
    ):
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_config = noise_config or {}
        self.use_real = use_real
        self.real_ratio = real_ratio
        self.real_data_dir = Path(real_data_dir) if real_data_dir else None
        
        self.synthetic_dataset = None
        self.real_dataset = None
        
    def setup(self, train_size: int = 8000, val_size: int = 1000):
        """Initialize datasets."""
        # Synthetic (always use)
        if self.dataset_name == 'ecg':
            self.synthetic_dataset = ECGDataset(
                seq_len=self.seq_len,
                num_samples=train_size,
                noise_config=self.noise_config
            )
        elif self.dataset_name == 'audio':
            self.synthetic_dataset = AudioDataset(
                seq_len=self.seq_len,
                num_samples=train_size,
                noise_config=self.noise_config
            )
        elif self.dataset_name == 'sensor':
            self.synthetic_dataset = SensorDataset(
                seq_len=self.seq_len,
                num_samples=train_size,
                noise_config=self.noise_config
            )
        
        # Real (optional)
        if self.use_real and self.real_data_dir:
            if self.dataset_name == 'ecg':
                self.real_dataset = PhysioNetECG(
                    data_dir=str(self.real_data_dir / 'ptb-xl'),
                    seq_len=self.seq_len,
                    noise_config=self.noise_config
                )
            elif self.dataset_name == 'audio':
                self.real_dataset = LibriSpeech(
                    data_dir=str(self.real_data_dir / 'librispeech'),
                    seq_len=self.seq_len,
                    noise_config=self.noise_config
                )
            elif self.dataset_name == 'sensor':
                self.real_dataset = UCIHARDataset(
                    data_dir=str(self.real_data_dir / 'uci-har'),
                    seq_len=self.seq_len,
                    noise_config=self.noise_config
                )
    
    def train_dataloader(self) -> DataLoader:
        """Get training loader (synthetic + optional real)."""
        if self.use_real and self.real_dataset and len(self.real_dataset) > 0:
            # Check if we want 100% real data
            if self.real_ratio >= 1.0:
                mixed_dataset = self.real_dataset
                print(f"🔥 Using 100% Real Data: {len(mixed_dataset)} records")
            else:
                # Mix synthetic and real
                from torch.utils.data import ConcatDataset

                # Calculate how many real samples to include
                total_samples = len(self.synthetic_dataset)
                real_samples = int(total_samples * self.real_ratio / (1 - self.real_ratio))

                # Subsample real dataset if needed
                if len(self.real_dataset) > real_samples:
                    real_subset = torch.utils.data.Subset(
                        self.real_dataset,
                        torch.randperm(len(self.real_dataset))[:real_samples].tolist()
                    )
                else:
                    real_subset = self.real_dataset

                mixed_dataset = ConcatDataset([self.synthetic_dataset, real_subset])
                print(f"🔀 Mixed dataset: {len(self.synthetic_dataset)} synthetic + {len(real_subset)} real")
        else:
            mixed_dataset = self.synthetic_dataset
            print(f"📊 Using synthetic data only: {len(self.synthetic_dataset)} samples")

        return DataLoader(
            mixed_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation loader (Real if available, else Synthetic)."""
        # If using real data, validate on real patients (Fold 10)
        if self.use_real and self.real_dataset and len(self.real_dataset) > 0:
            # Create a new instance for validation (Fold 10)
            if self.dataset_name == 'ecg':
                val_dataset = PhysioNetECG(
                    data_dir=str(self.real_data_dir / 'ptb-xl'),
                    seq_len=self.seq_len,
                    split='val',  # This loads Fold 10
                    noise_config=self.noise_config
                )
                print(f"🏥 Validating on {len(val_dataset)} real patient records")
            else:
                # Fallback to synthetic for non-ECG
                val_dataset = type(self.synthetic_dataset)(
                    seq_len=self.seq_len,
                    num_samples=1000,
                    noise_config=self.noise_config,
                    seed=1234
                )
        else:
            # Synthetic validation
            val_dataset = type(self.synthetic_dataset)(
                seq_len=self.seq_len,
                num_samples=1000,
                noise_config=self.noise_config,
                seed=1234
            )
            
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == "__main__":
    # ── Test data pipeline ───────────────────────────────────────────────────
    print("\n🧪 Testing Data Pipeline...\n")
    
    # Test noise injector
    injector = NoiseInjector(noise_type='mixed', snr_range=(5, 20))
    
    test_signal = np.sin(np.linspace(0, 4 * np.pi, 1024)).astype(np.float32)
    noisy = injector(test_signal)
    
    snr = 10 * np.log10(
        np.mean(test_signal ** 2) / np.mean((test_signal - noisy) ** 2)
    )
    print(f"Test signal SNR: {snr:.2f} dB")
    
    # Test dataset
    print("\nTesting ECG dataset...")
    ecg_dataset = ECGDataset(seq_len=1024, num_samples=100, noise_config={'noise_type': 'mixed'})
    
    noisy, clean, snr = ecg_dataset[0]
    print(f"Sample shape - Noisy: {noisy.shape}, Clean: {clean.shape}")
    print(f"Sample SNR: {snr:.2f} dB")
    
    print("\n✅ Data pipeline test passed!\n")
