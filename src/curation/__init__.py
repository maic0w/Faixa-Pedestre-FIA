"""Modulo de curadoria e amostragem de dados para anotacao."""
from .extract_frames import extract_controlled_frames
from .sample_extractor import extract_frames

__all__ = ["extract_controlled_frames", "extract_frames"]
