"""AstraLM: a small, inspectable decoder-only Transformer laboratory."""

from .bpe import BytePairTokenizer
from .config import ModelConfig
from .estimates import estimate
from .model import DecoderLM
from .tokenizer import ByteTokenizer

__all__ = ["BytePairTokenizer", "ByteTokenizer", "DecoderLM", "ModelConfig", "estimate"]
__version__ = "0.2.0"
