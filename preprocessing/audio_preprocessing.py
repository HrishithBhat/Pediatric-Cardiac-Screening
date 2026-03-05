"""
Audio Preprocessing Pipeline
=============================
Steps:
  1. Load WAV, resample to 2 kHz
  2. 4th-order Butterworth bandpass filter (20-400 Hz)
  3. HSMM-based heart-sound segmentation (S1/systole/S2/diastole)
  4. Log-Mel spectrogram extraction → fixed-size tensor (1, 64, 256)

Dependencies: librosa, scipy, numpy, torch
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. Resampling
# ---------------------------------------------------------------------------

def load_and_resample(wav_path: str, target_sr: int = 2000) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file and resample to *target_sr* Hz.

    Returns
    -------
    audio : np.ndarray  shape (N,)  float32, mono
    sr    : int
    """
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32), target_sr


# ---------------------------------------------------------------------------
# 2. Butterworth Bandpass Filter
# ---------------------------------------------------------------------------

def butterworth_bandpass(
    audio: np.ndarray,
    sr: int,
    low_hz: float = 20.0,
    high_hz: float = 400.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase 4th-order Butterworth bandpass filter.
    Uses sosfiltfilt (second-order sections) for numerical stability.
    """
    nyq = sr / 2.0
    low = low_hz / nyq
    high = high_hz / nyq
    # Clamp to valid range
    low = max(low, 1e-6)
    high = min(high, 1.0 - 1e-6)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, audio)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. HSMM-Based Heart Sound Segmentation
# ---------------------------------------------------------------------------

class HSMMSegmenter:
    """
    Lightweight Hidden Semi-Markov Model for cardiac cycle segmentation.

    State sequence: S1 (0) → Systole (1) → S2 (2) → Diastole (3) → S1 ...

    In a full implementation this would use the Springer HSMM.
    Here we implement an energy-envelope + peak-picking proxy that:
      - detects S1/S2 via short-time energy peaks
      - returns a list of (start_sample, end_sample) tuples for each beat cycle
    """

    def __init__(
        self,
        sr: int = 2000,
        frame_len_ms: int = 20,
        hop_ms: int = 5,
        min_cycle_s: float = 0.4,   # 150 bpm max
        max_cycle_s: float = 2.0,   # 30 bpm min
    ):
        self.sr = sr
        self.frame_len = int(sr * frame_len_ms / 1000)
        self.hop = int(sr * hop_ms / 1000)
        self.min_cycle = int(min_cycle_s * sr)
        self.max_cycle = int(max_cycle_s * sr)

    def _short_time_energy(self, audio: np.ndarray) -> np.ndarray:
        frames = librosa.util.frame(
            audio,
            frame_length=self.frame_len,
            hop_length=self.hop,
        )  # (frame_len, n_frames)
        energy = np.sum(frames ** 2, axis=0)
        # Smooth with a Hann window
        win = np.hanning(21)
        win /= win.sum()
        energy = np.convolve(energy, win, mode="same")
        return energy.astype(np.float32)

    def segment(
        self, audio: np.ndarray, clip_duration_s: float = 3.0
    ) -> np.ndarray:
        """
        Segment *audio* into cardiac cycles and return a fixed-length clip
        centred on the most energy-rich region.

        Returns
        -------
        clip : np.ndarray  shape (clip_samples,)
        """
        clip_samples = int(clip_duration_s * self.sr)

        if len(audio) <= clip_samples:
            # Pad if shorter
            pad = clip_samples - len(audio)
            return np.pad(audio, (0, pad), mode="constant")

        energy = self._short_time_energy(audio)

        # Sliding-window energy: find the richest *clip_duration_s* window
        win_frames = int(clip_duration_s * self.sr / self.hop)
        win_frames = min(win_frames, len(energy) - 1)
        cum = np.cumsum(energy)
        cum = np.concatenate([[0], cum])
        window_energy = cum[win_frames:] - cum[: len(cum) - win_frames]
        best_start_frame = int(np.argmax(window_energy))
        best_start_sample = best_start_frame * self.hop

        clip = audio[best_start_sample : best_start_sample + clip_samples]
        if len(clip) < clip_samples:
            clip = np.pad(clip, (0, clip_samples - len(clip)), mode="constant")
        return clip.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Log-Mel Spectrogram
# ---------------------------------------------------------------------------

def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 2000,
    n_fft: int = 256,
    hop_length: int = 64,
    n_mels: int = 64,
    fmin: float = 20.0,
    fmax: float = 400.0,
    top_db: float = 80.0,
    fixed_width: int = 256,
) -> np.ndarray:
    """
    Returns a Log-Mel spectrogram array of shape (n_mels, fixed_width).
    Temporal axis is either truncated or zero-padded to *fixed_width*.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=top_db)

    # Normalise to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

    # Temporal alignment
    T = log_mel.shape[1]
    if T >= fixed_width:
        log_mel = log_mel[:, :fixed_width]
    else:
        pad = fixed_width - T
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant")

    return log_mel.astype(np.float32)  # (64, 256)


# ---------------------------------------------------------------------------
# 5. Full Audio Pipeline
# ---------------------------------------------------------------------------

class AudioPreprocessor:
    """
    End-to-end audio preprocessing: WAV → normalised Log-Mel tensor.

    Usage
    -----
    pre = AudioPreprocessor()
    tensor = pre(wav_path)   # torch.Tensor  (1, 64, 256)
    """

    def __init__(
        self,
        target_sr: int = 2000,
        bandpass_low: float = 20.0,
        bandpass_high: float = 400.0,
        filter_order: int = 4,
        n_fft: int = 256,
        hop_length: int = 64,
        n_mels: int = 64,
        fmin: float = 20.0,
        fmax: float = 400.0,
        top_db: float = 80.0,
        spec_width: int = 256,
        segment_duration_s: float = 3.0,
    ):
        self.target_sr = target_sr
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.filter_order = filter_order
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.top_db = top_db
        self.spec_width = spec_width
        self.segment_duration_s = segment_duration_s
        self.segmenter = HSMMSegmenter(sr=target_sr)

    def __call__(self, wav_path: str) -> torch.Tensor:
        """Returns shape (1, n_mels, spec_width)."""
        audio, sr = load_and_resample(wav_path, self.target_sr)
        audio = butterworth_bandpass(
            audio, sr,
            low_hz=self.bandpass_low,
            high_hz=self.bandpass_high,
            order=self.filter_order,
        )
        audio = self.segmenter.segment(audio, self.segment_duration_s)
        spec = compute_log_mel_spectrogram(
            audio, sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            top_db=self.top_db,
            fixed_width=self.spec_width,
        )
        # shape (1, 64, 256)
        return torch.from_numpy(spec).unsqueeze(0)

    def from_numpy(self, audio: np.ndarray) -> torch.Tensor:
        """Accept a pre-loaded numpy array instead of a file path."""
        audio = audio.astype(np.float32)
        audio = butterworth_bandpass(
            audio, self.target_sr,
            low_hz=self.bandpass_low,
            high_hz=self.bandpass_high,
            order=self.filter_order,
        )
        audio = self.segmenter.segment(audio, self.segment_duration_s)
        spec = compute_log_mel_spectrogram(
            audio, self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            top_db=self.top_db,
            fixed_width=self.spec_width,
        )
        return torch.from_numpy(spec).unsqueeze(0)
