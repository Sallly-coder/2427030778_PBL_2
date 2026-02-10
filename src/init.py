"""
src/ — Speech Emotion Recognition module
"""
from src.audio_processing import AudioProcessor
from src.feature_extraction import FeatureExtractor
from src.model import EmotionClassifier

__all__ = ["AudioProcessor", "FeatureExtractor", "EmotionClassifier"]
