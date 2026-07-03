"""
Modelos de datos para el PCM
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json


@dataclass
class CompressionResult:
    """Resultado de una operación de compresión de prompt"""
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    processing_time_ms: float
    target_model: Optional[str] = None
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para JSON"""
        return {
            "original_prompt": self.original_prompt,
            "compressed_prompt": self.compressed_prompt,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": round(self.compression_ratio, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "target_model": self.target_model,
            "strategy": self.strategy,
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convierte a JSON"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class CompressionMetrics:
    """Métricas de compresión para múltiples prompts"""
    total_prompts: int = 0
    avg_compression_ratio: float = 0.0
    avg_processing_time_ms: float = 0.0
    min_ratio: float = float('inf')
    max_ratio: float = 0.0
    total_tokens_saved: int = 0
    results: List[CompressionResult] = field(default_factory=list)
    
    def add_result(self, result: CompressionResult) -> None:
        """Añade un resultado a las métricas"""
        self.results.append(result)
        self.total_prompts += 1
        self.total_tokens_saved += result.original_tokens - result.compressed_tokens
        
        # Actualiza promedios
        ratios = [r.compression_ratio for r in self.results]
        self.avg_compression_ratio = sum(ratios) / len(ratios)
        self.min_ratio = min(self.min_ratio, result.compression_ratio)
        self.max_ratio = max(self.max_ratio, result.compression_ratio)
        
        times = [r.processing_time_ms for r in self.results]
        self.avg_processing_time_ms = sum(times) / len(times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "total_prompts": self.total_prompts,
            "avg_compression_ratio": round(self.avg_compression_ratio, 4),
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "min_compression_ratio": round(self.min_ratio, 4),
            "max_compression_ratio": round(self.max_ratio, 4),
            "total_tokens_saved": self.total_tokens_saved
        }


@dataclass  
class ComparisonResult:
    """Resultado de comparación entre prompt original y comprimido"""
    original: str
    compressed: str
    semantic_similarity: float  # 0.0 a 1.0
    token_savings: int
    compression_ratio: float
    evaluation: str  # "excellent", "good", "fair", "poor"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "compressed": self.compressed,
            "semantic_similarity": round(self.semantic_similarity, 4),
            "token_savings": self.token_savings,
            "compression_ratio": round(self.compression_ratio, 4),
            "evaluation": self.evaluation
        }
