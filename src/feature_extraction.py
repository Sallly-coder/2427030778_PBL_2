"""
feature_extraction.py
======================
Computes audio features (MFCC, pitch, energy, ZCR) from preprocessed signals.
Returns feature vectors suitable for scikit-learn classifiers.

Usage:
    from src.feature_extraction import FeatureExtractor

    extractor = FeatureExtractor(n_mfcc=40)
    features  = extractor.extract_all(signal, sr)   # → 1-D numpy array
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class FeatureExtractor:
    """
    Extracts audio features from a pre-processed signal.

    Parameters
    ----------
    n_mfcc   : int   Number of MFCC coefficients to compute (default: 40)
    hop_length: int  Hop length for STFT frames (default: 512)
    n_fft    : int   FFT window size (default: 2048)
    """

    def __init__(self, n_mfcc: int = 40,
                 hop_length: int = 512,
                 n_fft: int = 2048):
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft

    # ------------------------------------------------------------------
    # Individual feature methods
    # ------------------------------------------------------------------

    def extract_mfcc(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute MFCCs and summarise each coefficient as [mean, std].

        Returns
        -------
        features : np.ndarray of shape (2 * n_mfcc,)
            First n_mfcc values = mean per coefficient
            Last  n_mfcc values = std  per coefficient
        """
        mfcc = librosa.feature.mfcc(y=signal, sr=sr,
                                     n_mfcc=self.n_mfcc,
                                     n_fft=self.n_fft,
                                     hop_length=self.hop_length)
        # mfcc shape: (n_mfcc, n_frames)
        return np.hstack([np.mean(mfcc, axis=1),
                          np.std(mfcc, axis=1)])

    def extract_mfcc_delta(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute MFCC + delta + delta-delta (captures temporal dynamics).

        Returns
        -------
        features : np.ndarray of shape (6 * n_mfcc,)
        """
        mfcc = librosa.feature.mfcc(y=signal, sr=sr,
                                     n_mfcc=self.n_mfcc,
                                     n_fft=self.n_fft,
                                     hop_length=self.hop_length)
        delta    = librosa.feature.delta(mfcc)
        delta2   = librosa.feature.delta(mfcc, order=2)

        all_feat = np.vstack([mfcc, delta, delta2])  # (3*n_mfcc, n_frames)
        return np.hstack([np.mean(all_feat, axis=1),
                          np.std(all_feat, axis=1)])

    def extract_pitch(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Estimate fundamental frequency (F0 / pitch) using librosa.yin.

        Returns
        -------
        features : np.ndarray of shape (3,)  [mean, std, median pitch]
        """
        f0 = librosa.yin(signal, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[f0 > 0]   # remove unvoiced frames (0 Hz)
        if len(f0_clean) == 0:
            return np.zeros(3)
        return np.array([
            np.mean(f0_clean),
            np.std(f0_clean),
            np.median(f0_clean)
        ])

    def extract_energy(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute RMS energy frame-by-frame.

        Returns
        -------
        features : np.ndarray of shape (2,)  [mean, std RMS energy]
        """
        rms = librosa.feature.rms(y=signal,
                                   hop_length=self.hop_length)[0]
        return np.array([np.mean(rms), np.std(rms)])

    def extract_zcr(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute Zero Crossing Rate — captures noisiness / consonants.

        Returns
        -------
        features : np.ndarray of shape (2,)  [mean, std ZCR]
        """
        zcr = librosa.feature.zero_crossing_rate(
            y=signal, hop_length=self.hop_length)[0]
        return np.array([np.mean(zcr), np.std(zcr)])

    def extract_chroma(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Chroma features: pitch class profiles (12 bins).

        Returns
        -------
        features : np.ndarray of shape (24,)  [mean, std per bin]
        """
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr,
                                              n_fft=self.n_fft,
                                              hop_length=self.hop_length)
        return np.hstack([np.mean(chroma, axis=1),
                          np.std(chroma, axis=1)])

    def extract_mel_spectrogram(self, signal: np.ndarray,
                                 sr: int, n_mels: int = 32) -> np.ndarray:
        """
        Log Mel-spectrogram mean+std (compact version).

        Returns
        -------
        features : np.ndarray of shape (2 * n_mels,)
        """
        mel = librosa.feature.melspectrogram(y=signal, sr=sr,
                                              n_mels=n_mels,
                                              n_fft=self.n_fft,
                                              hop_length=self.hop_length)
        log_mel = librosa.power_to_db(mel)
        return np.hstack([np.mean(log_mel, axis=1),
                          np.std(log_mel, axis=1)])

    # ------------------------------------------------------------------
    # Combined feature vector
    # ------------------------------------------------------------------

    def extract_all(self, signal: np.ndarray, sr: int,
                    use_delta: bool = True,
                    use_pitch: bool = True,
                    use_energy: bool = True,
                    use_zcr: bool = True,
                    use_chroma: bool = False) -> np.ndarray:
        """
        Concatenate selected features into a single 1-D vector.

        Default combination (use_delta=True, rest default):
            MFCC-delta (6*40=240) + pitch (3) + energy (2) + zcr (2)
            = 247-dimensional vector

        Parameters
        ----------
        signal      : np.ndarray   pre-processed audio signal
        sr          : int          sample rate
        use_delta   : bool         use MFCC + delta + delta2 (richer)
        use_pitch   : bool
        use_energy  : bool
        use_zcr     : bool
        use_chroma  : bool

        Returns
        -------
        features : np.ndarray (1-D)
        """
        parts = []

        if use_delta:
            parts.append(self.extract_mfcc_delta(signal, sr))
        else:
            parts.append(self.extract_mfcc(signal, sr))

        if use_pitch:
            parts.append(self.extract_pitch(signal, sr))

        if use_energy:
            parts.append(self.extract_energy(signal))

        if use_zcr:
            parts.append(self.extract_zcr(signal))

        if use_chroma:
            parts.append(self.extract_chroma(signal, sr))

        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def plot_mfcc(self, signal: np.ndarray, sr: int,
                  title: str = "MFCC", save_path: str = None):
        """Plot MFCC heatmap."""
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=self.n_mfcc)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis="time",
                                  sr=sr, hop_length=self.hop_length,
                                  cmap="coolwarm")
        plt.colorbar()
        plt.title(title)
        plt.ylabel("MFCC Coefficient")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_mel_spectrogram(self, signal: np.ndarray, sr: int,
                              title: str = "Mel Spectrogram",
                              save_path: str = None):
        """Plot log Mel-spectrogram."""
        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel, x_axis="time", y_axis="mel",
                                  sr=sr, cmap="viridis")
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
    from src.audio_processing import AudioProcessor

    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <path_to_wav>")
        sys.exit(1)

    wav_path = sys.argv[1]

    proc = AudioProcessor()
    signal, sr = proc.full_preprocess(wav_path)

    ext = FeatureExtractor(n_mfcc=40)
    features = ext.extract_all(signal, sr)

    print(f"Feature vector shape : {features.shape}")
    print(f"First 10 values      : {features[:10].round(4)}")

    ext.plot_mfcc(signal, sr)
    ext.plot_mel_spectrogram(signal, sr)
