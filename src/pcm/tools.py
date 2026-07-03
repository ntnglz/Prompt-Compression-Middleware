"""
Herramientas compartidas del servidor PCM (HTTP y MCP).
"""

import logging
import time
from typing import List, Optional

from .compressor import PromptCompressor
from .models import ComparisonResult, CompressionMetrics, CompressionResult

logger = logging.getLogger(__name__)


class PCMServerTools:
    """Implementación de las herramientas expuestas por HTTP y MCP."""

    def __init__(self, compressor: PromptCompressor):
        self.compressor = compressor

    @property
    def model_name(self) -> str:
        return self.compressor.config.model

    async def compress_prompt(
        self,
        prompt: str,
        strategy: Optional[str] = None,
        target_model: Optional[str] = None,
    ) -> dict:
        """Comprime un prompt preservando su intención semántica."""
        logger.info("Comprimiendo prompt: %s...", prompt[:50])

        start_time = time.time()
        result: CompressionResult = self.compressor.compress(
            prompt=prompt,
            strategy=strategy,
            target_model=target_model,
        )
        processing_time = time.time() - start_time

        logger.info(
            "Compresión completada en %.2fs - Ratio: %.2f%%",
            processing_time,
            result.compression_ratio * 100,
        )
        return result.to_dict()

    async def batch_compress(
        self,
        prompts: List[str],
        strategy: Optional[str] = None,
    ) -> dict:
        """Comprime múltiples prompts en batch."""
        logger.info("Comprimiendo %d prompts en batch", len(prompts))

        start_time = time.time()
        results = self.compressor.batch_compress(prompts, strategy=strategy)
        batch_time = time.time() - start_time

        total_original = sum(r.original_tokens for r in results)
        total_compressed = sum(r.compressed_tokens for r in results)
        avg_ratio = sum(r.compression_ratio for r in results) / len(results) if results else 0

        logger.info("Batch completado en %.2fs - Ratio promedio: %.2f%%", batch_time, avg_ratio * 100)

        return {
            "results": [r.to_dict() for r in results],
            "statistics": {
                "total_prompts": len(results),
                "total_original_tokens": total_original,
                "total_compressed_tokens": total_compressed,
                "avg_compression_ratio": round(avg_ratio, 4),
                "total_processing_time": round(batch_time, 2),
            },
        }

    async def compare_prompts(
        self,
        original: str,
        compressed: Optional[str] = None,
    ) -> dict:
        """Compara un prompt original con su versión comprimida."""
        logger.info("Comparando prompts...")

        result: ComparisonResult = self.compressor.compare_prompts(
            original=original,
            compressed=compressed,
        )

        logger.info(
            "Similitud semántica: %.2f%% - Evaluación: %s",
            result.semantic_similarity * 100,
            result.evaluation,
        )
        return result.to_dict()

    async def estimate_tokens(self, text: str) -> dict:
        """Estima el número de tokens de un texto."""
        token_count = self.compressor._count_tokens(text, self.model_name)
        return {
            "text": text,
            "tokens": token_count,
        }

    async def get_compression_stats(self, prompts: List[str]) -> dict:
        """Obtiene estadísticas de compresión para una lista de prompts."""
        metrics = CompressionMetrics()

        for prompt in prompts:
            result = self.compressor.compress(prompt)
            metrics.add_result(result)

        return metrics.to_dict()

    async def health_check(self) -> dict:
        """Verifica el estado del servidor y el modelo."""
        try:
            import ollama

            models = ollama.list()
            available = [m.get("model", m.get("name", "")) for m in models.get("models", [])]
            model_available = any(
                self.model_name in name or name.startswith(f"{self.model_name}:")
                for name in available
            )

            if not model_available:
                return {
                    "status": "unhealthy",
                    "error": f"Modelo '{self.model_name}' no disponible en Ollama",
                    "available_models": available,
                    "timestamp": time.time(),
                }

            return {
                "status": "healthy",
                "model": self.model_name,
                "version": "0.1.0",
                "timestamp": time.time(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
            }
