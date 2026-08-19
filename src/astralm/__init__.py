"""AstraLM: a small, inspectable decoder-only Transformer laboratory."""

from .config import ModelConfig
from .model import DecoderLM
from .tokenizer import ByteTokenizer

__all__ = ["ByteTokenizer", "DecoderLM", "ModelConfig"]
__version__ = "0.1.0"

