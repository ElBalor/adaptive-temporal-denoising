"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ADAPTIVE TEMPORAL DENOISING AUTOENCODER                                     ║
║  Experiment 01 - The Wheelchair Bait                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .model import AdaptiveTemporalDenoisingAE, TCNBlock, MultiScaleBranch
from .data import NoiseInjector, ECGDataset, AudioDataset, SensorDataset
from .train import AdaptiveDenoisingTrainer, CompositeDenoisingLoss

__version__ = "0.1.0"
__author__ = "The Digital Necromancer"

__all__ = [
    "AdaptiveTemporalDenoisingAE",
    "TCNBlock",
    "MultiScaleBranch",
    "NoiseInjector",
    "ECGDataset",
    "AudioDataset",
    "SensorDataset",
    "AdaptiveDenoisingTrainer",
    "CompositeDenoisingLoss",
]
