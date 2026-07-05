"""
Prompt Compression Middleware (PCM)

Un servidor MCP para comprimir prompts de manera semántica,
optimizando la comunicación con LLM.
"""

__version__ = "1.0.0"
__author__ = "Antonio J. Gonzalez"

from .compressor import PromptCompressor
from .models import CompressionResult, CompressionMetrics
from .tools import PCMServerTools

__all__ = [
    "PromptCompressor",
    "CompressionResult",
    "CompressionMetrics",
    "PCMServerTools",
]
