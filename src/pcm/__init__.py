"""
Prompt Compression Middleware (PCM)

Un servidor MCP para comprimir prompts de manera semántica,
optimizando la comunicación con LLM.
"""

__version__ = "1.0.0"
__author__ = "Antonio J. Gonzalez"

from .compressor import PromptCompressor
from .message_assembly import build_proxy_system_prompt, build_system_prompt
from .models import CompressionResult, CompressionMetrics
from .output_directives import build_output_directives, has_response_block
from .tools import PCMServerTools
from .turn_cost import TurnCostMetrics, compute_turn_cost

__all__ = [
    "PromptCompressor",
    "CompressionResult",
    "CompressionMetrics",
    "PCMServerTools",
    "build_output_directives",
    "has_response_block",
    "build_system_prompt",
    "build_proxy_system_prompt",
    "TurnCostMetrics",
    "compute_turn_cost",
]
