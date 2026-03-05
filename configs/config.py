"""
Central configuration for the Pediatric Cardiac Screening System.
All hyperparameters, paths, and model settings are defined here.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Audio Configuration
# ---------------------------------------------------------------------------
@dataclass
class AudioConfig:
    # Raw waveform settings
    original_sr: int = 44100          # Source sample rate
    target_sr: int = 2000             # Downsample to 2 kHz (per spec)

    # Butterworth bandpass filter (4th-order, 20-400 Hz)
    bandpass_low: float = 20.0
    bandpass_high: float = 400.0
    filter_order: int = 4

    # Log-Mel spectrogram settings
    n_fft: int = 256
    hop_length: int = 64
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 400.0
    top_db: float = 80.0              # Dynamic range for amplitude->dB

    # CRNN input shape  (C, H, W) → (1, 64, T)
    spec_height: int = 64
    spec_width: int = 256             # fixed temporal width via padding/truncation

    # HSMM segmentation
    hsmm_n_states: int = 4            # S1, systole, S2, diastole
    segment_duration_s: float = 3.0   # clip length after segmentation

    # CRNN architecture
    lstm_hidden: int = 256
    lstm_layers: int = 2
    crnn_embed_dim: int = 512


# ---------------------------------------------------------------------------
# Image Configuration (shared base)
# ---------------------------------------------------------------------------
@dataclass
class ImageConfig:
    # Aspect-ratio-preserving padding target
    image_size: Tuple[int, int] = (224, 224)

    # CLAHE
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # Z-score normalization (ImageNet stats – good starting point)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Ultrasound Configuration (NTS-Net)
# ---------------------------------------------------------------------------
@dataclass
class UltrasoundConfig(ImageConfig):
    nts_num_parts: int = 6            # Navigator proposal regions
    nts_top_k: int = 3                # Top-K scrutinized parts
    nts_embed_dim: int = 512
    nts_num_classes: int = 1          # Binary after "surgery"


# ---------------------------------------------------------------------------
# X-Ray Configuration (EfficientNetV2-S)
# ---------------------------------------------------------------------------
@dataclass
class XRayConfig(ImageConfig):
    efficientnet_embed_dim: int = 512


# ---------------------------------------------------------------------------
# GMU / Fusion Configuration
# ---------------------------------------------------------------------------
@dataclass
class FusionConfig:
    audio_embed_dim: int = 512
    us_embed_dim: int = 512
    xray_embed_dim: int = 512
    gmu_hidden_dim: int = 512
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    mlp_dropout: float = 0.5
    num_classes: int = 1              # Binary: 0=Pass, 1=Refer

    # Modality dropout probabilities during training
    modality_drop_p: float = 0.2      # probability to zero-out a modality embedding


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Phase-1 specialist training
    specialist_epochs: int = 50
    specialist_lr: float = 1e-4
    specialist_batch_size: int = 32
    specialist_weight_decay: float = 1e-4

    # Phase-2 GMU training
    gmu_epochs: int = 30
    gmu_lr: float = 5e-5
    gmu_batch_size: int = 16
    gmu_weight_decay: float = 1e-4

    # Scheduler
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5

    # Misc
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"              # set to "cpu" if no GPU
    amp: bool = True                  # automatic mixed precision

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


# ---------------------------------------------------------------------------
# Assembled master config
# ---------------------------------------------------------------------------
@dataclass
class MasterConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    ultrasound: UltrasoundConfig = field(default_factory=UltrasoundConfig)
    xray: XRayConfig = field(default_factory=XRayConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Singleton – import this everywhere
CFG = MasterConfig()
