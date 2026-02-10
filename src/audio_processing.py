"""
audio_processing.py
====================
Handles audio file loading, normalization, and basic denoising.

Usage:
    from src.audio_processing import AudioProcessor

    processor = AudioProcessor(sample_rate=22050)
    signal, sr = processor.load("path/to/file.wav")
    signal = processor.normalize(signal)
"""

import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt


class AudioProcessor:
    """
    Handles loading and preprocessing of audio (.wav) files.

    Parameters
    ----------
    sample_rate : int
        Target sample rate to resample all audio to (default: 22050 Hz).
    duration : float or None
        If set, clips/pads all audio to this duration in seconds.
    """

    def __init__(self, sample_rate: int = 22050, duration: float = 3.0):
        self.sample_rate = sample_rate
        self.duration = duration

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Load a .wav file and resample to self.sample_rate.

        Parameters
        ----------
        file_path : str
            Path to the .wav audio file.

        Returns
        -------
        signal : np.ndarray  (1-D float array)
        sr     : int         (sample rate)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        signal, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return signal, sr

    def load_fixed_length(self, file_path: str) -> np.ndarray:
        """
        Load and pad/trim audio to a fixed number of samples
        (self.duration * self.sample_rate).

        Returns
        -------
        signal : np.ndarray of shape (n_samples,)
        """
        signal, _ = self.load(file_path)
        max_samples = int(self.duration * self.sample_rate)
        signal = librosa.util.fix_length(signal, size=max_samples)
        return signal

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal to the range [-1, 1].

        Parameters
        ----------
        signal : np.ndarray

        Returns
        -------
        normalized : np.ndarray
        """
        max_val = np.max(np.abs(signal))
        if max_val == 0:
            return signal  # avoid divide-by-zero on silent clips
        return signal / max_val

    def remove_silence(self, signal: np.ndarray,
                        top_db: int = 20) -> np.ndarray:
        """
        Trim leading and trailing silence from a signal.

        Parameters
        ----------
        signal : np.ndarray
        top_db : int
            Threshold (in dB below peak) below which frames are trimmed.

        Returns
        -------
        trimmed : np.ndarray
        """
        trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
        return trimmed

    def apply_preemphasis(self, signal: np.ndarray,
                          coeff: float = 0.97) -> np.ndarray:
        """
        Apply pre-emphasis filter to boost high frequencies.
        Commonly used before MFCC extraction.

        y[t] = x[t] - coeff * x[t-1]

        Parameters
        ----------
        signal : np.ndarray
        coeff  : float  (default 0.97)

        Returns
        -------
        emphasized : np.ndarray
        """
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    def full_preprocess(self, file_path: str,
                        use_preemphasis: bool = True) -> tuple[np.ndarray, int]:
        """
        Full preprocessing pipeline:
            load → trim silence → normalize → (optional) pre-emphasis

        Parameters
        ----------
        file_path       : str
        use_preemphasis : bool

        Returns
        -------
        signal : np.ndarray
        sr     : int
        """
        signal, sr = self.load(file_path)
        signal = self.remove_silence(signal)
        signal = self.normalize(signal)
        if use_preemphasis:
            signal = self.apply_preemphasis(signal)
        return signal, sr

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def plot_waveform(self, signal: np.ndarray, title: str = "Waveform",
                      save_path: str = None):
        """Plot time-domain waveform."""
        plt.figure(figsize=(10, 3))
        times = np.linspace(0, len(signal) / self.sample_rate, len(signal))
        plt.plot(times, signal, color="royalblue", linewidth=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_spectrogram(self, signal: np.ndarray,
                         title: str = "Spectrogram", save_path: str = None):
        """Plot log-power spectrogram."""
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(signal)), ref=np.max
        )
        librosa.display.specshow(D, sr=self.sample_rate,
                                 x_axis="time", y_axis="hz",
                                 cmap="magma")
        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


# ------------------------------------------------------------------
# Quick standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_processing.py <path_to_wav>")
        sys.exit(1)

    wav_path = sys.argv[1]
    proc = AudioProcessor(sample_rate=22050, duration=3.0)

    print(f"Loading: {wav_path}")
    signal, sr = proc.full_preprocess(wav_path)
    print(f"  Sample rate : {sr} Hz")
    print(f"  Duration    : {len(signal)/sr:.2f} s")
    print(f"  Min/Max     : {signal.min():.4f} / {signal.max():.4f}")

    proc.plot_waveform(signal, title=os.path.basename(wav_path))
    proc.plot_spectrogram(signal, title=os.path.basename(wav_path))
