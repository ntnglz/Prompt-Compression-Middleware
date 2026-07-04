"""
Compresor de prompts para PCM

Este módulo contiene la lógica principal de compresión de prompts.
"""

import re
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import ollama

from .models import CompressionResult, ComparisonResult
from .compression_policy import CompressionPolicy
from .compression_prompts import COMPRESSION_SYSTEM_PROMPT, PCM_SYSTEM_FULL, PCM_SYSTEM_GLOSSARY

# Configura logging
logger = logging.getLogger(__name__)

@dataclass
class CompressorConfig:
    """Configuración del compresor"""
    model: str = "granite4.1:3b"
    temperature: float = 0.1
    timeout: int = 120
    max_tokens: Optional[int] = None
    strategy: str = "balanced"  # "aggressive", "balanced", "conservative"
    target_model: Optional[str] = None  # Modelos destino: "gpt-4", "claude-3", etc.
    think: Optional[bool] = None  # None = auto (Qwen3 requiere thinking para PCM)
    evaluator_model: str = "granite4.1:3b"  # Evaluador fijo para comparativas justas
    min_instruction_tokens: int = 12  # No comprimir instrucciones más cortas
    glossary_only: bool = False  # True → PCM_SYSTEM_GLOSSARY (sin few-shots)


class PromptCompressor:
    """
    Clase principal para comprimir prompts usando LLM.
    
    Uso:
        compressor = PromptCompressor()
        result = compressor.compress("Analiza este código...")
        print(result.compressed_prompt)
    """
    
    def __init__(self, config: Optional[CompressorConfig] = None):
        """
        Inicializa el compresor.
        
        Args:
            config: Configuración del compresor. Si None, usa valores por defecto.
        """
        self.config = config or CompressorConfig()
        self._validate_config()
        logger.info(f"Compresor inicializado con modelo: {self.config.model}")
    
    def _validate_config(self) -> None:
        """Valida la configuración"""
        if self.config.temperature < 0 or self.config.temperature > 1:
            raise ValueError("Temperature debe estar entre 0 y 1")
        
        if self.config.timeout <= 0:
            raise ValueError("Timeout debe ser positivo")

    def _is_qwen3_model(self, model: Optional[str] = None) -> bool:
        return "qwen3" in (model or self.config.model).lower()

    def _resolve_think(self, model: Optional[str] = None) -> Optional[bool]:
        """Qwen3 necesita thinking para seguir el formato PCM; otros modelos no lo usan."""
        if self.config.think is not None:
            return self.config.think
        if self._is_qwen3_model(model):
            return True
        return None

    def _extract_pcm_line(self, text: str) -> str:
        """Extrae la línea PCM (TASK=...) de respuestas ruidosas."""
        for line in text.splitlines():
            candidate = line.strip()
            if candidate.startswith("TASK="):
                return candidate
        match = re.search(r"TASK=[^\n]+", text)
        return match.group(0).strip() if match else text.strip()

    def _parse_semantic_score(self, raw: str) -> float:
        """Parsea un score 0-1 aunque el modelo añada texto extra."""
        cleaned = raw.strip()
        try:
            return float(cleaned)
        except ValueError:
            pass
        matches = re.findall(r"0?\.\d+|1\.0|1|0", cleaned)
        if matches:
            return float(matches[-1])
        raise ValueError(f"No se pudo parsear score semántico: {raw[:120]!r}")

    def _ollama_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        think: Optional[bool] = None,
        num_predict: Optional[int] = None,
    ) -> dict[str, Any]:
        """Llama a Ollama aplicando política de thinking por modelo."""
        effective_model = model or self.config.model
        options: dict[str, Any] = {
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if num_predict is not None:
            options["num_predict"] = num_predict

        chat_kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "options": options,
        }
        effective_think = self._resolve_think(effective_model) if think is None else think
        if effective_think is not None:
            chat_kwargs["think"] = effective_think
        return ollama.chat(**chat_kwargs)
    
    def _get_compression_prompt(self, strategy: str) -> str:
        """Obtiene el prompt de compresión según la estrategia"""
        base_prompt = PCM_SYSTEM_GLOSSARY if self.config.glossary_only else PCM_SYSTEM_FULL
        
        # Ajusta el prompt según la estrategia
        if strategy == "aggressive":
            base_prompt += "\n\nENFATIZA: Reducción máxima de tokens. Acepta pequeña pérdida de naturalidad."
        elif strategy == "conservative":
            base_prompt += "\n\nENFATIZA: Preserva cada detalle semántico. Reducción moderada de tokens."
        
        return base_prompt
    
    def _count_tokens(self, text: str, model: str = "granite4.1:3b") -> int:
        """Cuenta tokens usando el tokenizador de Ollama"""
        try:
            # Ollama no tiene método tokenize directo en su API Python
            # Usamos tiktoken como alternativa (requiere pip install tiktoken)
            try:
                import tiktoken
                enc = tiktoken.encoding_for_model("gpt-4")
                return len(enc.encode(text))
            except ImportError:
                # Usar estimación simple
                logger.warning("tiktoken no disponible. Usando estimación simple.")
                return len(text) // 4
        except Exception as e:
            logger.warning(f"Error al contar tokens: {e}. Usando longitud de caracteres / 4")
            return len(text) // 4  # Estimación aproximada
    
    def compress(
        self, 
        prompt: str,
        strategy: Optional[str] = None,
        target_model: Optional[str] = None
    ) -> CompressionResult:
        """
        Comprime un prompt.
        
        Args:
            prompt: El prompt original a comprimir
            strategy: Estrategia de compresión ("aggressive", "balanced", "conservative")
            target_model: Modelo destino para optimizar la compresión
            
        Returns:
            CompressionResult con el prompt comprimido y métricas
        """
        start_time = time.time()
        
        # Usa configuración personalizada o por defecto
        effective_strategy = strategy or self.config.strategy
        effective_target = target_model or self.config.target_model
        
        # Contar tokens originales
        original_tokens = self._count_tokens(prompt, self.config.model)
        logger.info(f"Prompt original: {original_tokens} tokens")

        policy = CompressionPolicy(
            min_instruction_tokens=self.config.min_instruction_tokens
        )
        count_tokens = lambda text: self._count_tokens(text, self.config.model)

        should_compress, skip_reason = policy.should_compress_instruction(
            prompt, count_tokens
        )
        if not should_compress:
            logger.info("Compresión omitida: %s", skip_reason)
            return self._passthrough_result(
                prompt=prompt,
                original_tokens=original_tokens,
                start_time=start_time,
                strategy=effective_strategy,
                target_model=effective_target,
                skip_reason=skip_reason,
            )
        
        # Generar prompt de compresión
        compression_prompt = self._get_compression_prompt(effective_strategy)
        
        # Añadir contexto del modelo destino si se especifica
        if effective_target:
            compression_prompt += f"\n\nModelo destino: {effective_target}"
        
        try:
            response = self._ollama_chat(
                messages=[
                    {"role": "system", "content": compression_prompt},
                    {"role": "user", "content": f"Prompt a comprimir:\n{prompt}"}
                ],
            )
            
            compressed_prompt = self._extract_pcm_line(
                response["message"]["content"].strip()
            )
            
            # Contar tokens comprimidos
            compressed_tokens = self._count_tokens(compressed_prompt, self.config.model)

            should_apply, revert_reason = policy.should_apply_compression(
                prompt,
                compressed_prompt,
                count_tokens,
            )
            if not should_apply:
                logger.info("Compresión revertida: %s", revert_reason)
                return self._passthrough_result(
                    prompt=prompt,
                    original_tokens=original_tokens,
                    start_time=start_time,
                    strategy=effective_strategy,
                    target_model=effective_target,
                    skip_reason=revert_reason,
                    attempted_pcm=compressed_prompt,
                )
            
            # Calcular ratio
            compression_ratio = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0
            processing_time = (time.time() - start_time) * 1000  # en ms
            
            logger.info(f"Prompt comprimido: {compressed_tokens} tokens")
            logger.info(f"Ratio de compresión: {compression_ratio:.2%}")
            
            metadata = {
                "model_used": self.config.model,
                "temperature": self.config.temperature,
                "think": self._resolve_think(),
            }

            return CompressionResult(
                original_prompt=prompt,
                compressed_prompt=compressed_prompt,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=compression_ratio,
                processing_time_ms=processing_time,
                target_model=effective_target,
                strategy=effective_strategy,
                metadata=metadata,
            )
            
        except Exception as e:
            logger.error(f"Error al comprimir prompt: {e}")
            raise RuntimeError(f"Error de compresión: {str(e)}")

    def _passthrough_result(
        self,
        *,
        prompt: str,
        original_tokens: int,
        start_time: float,
        strategy: str,
        target_model: Optional[str],
        skip_reason: str,
        attempted_pcm: Optional[str] = None,
    ) -> CompressionResult:
        """Devuelve el prompt original sin comprimir cuando no hay ahorro neto."""
        metadata: Dict[str, Any] = {
            "model_used": self.config.model,
            "temperature": self.config.temperature,
            "think": self._resolve_think(),
            "skipped": True,
            "skip_reason": skip_reason,
        }
        if attempted_pcm:
            metadata["attempted_pcm"] = attempted_pcm

        return CompressionResult(
            original_prompt=prompt,
            compressed_prompt=prompt,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=0.0,
            processing_time_ms=(time.time() - start_time) * 1000,
            target_model=target_model,
            strategy=strategy,
            metadata=metadata,
        )
    
    def compare_prompts(
        self,
        original: str,
        compressed: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compara un prompt original con su versión comprimida.
        
        Args:
            original: Prompt original
            compressed: Prompt comprimido (si None, lo comprime primero)
            strategy: Estrategia para comprimir si compressed es None
            
        Returns:
            ComparisonResult con métricas de comparación
        """
        if compressed is None:
            # Comprimir si no se proporciona
            result = self.compress(original, strategy=strategy)
            compressed = result.compressed_prompt
        
        # Contar tokens
        orig_tokens = self._count_tokens(original, self.config.model)
        comp_tokens = self._count_tokens(compressed, self.config.model)
        token_savings = orig_tokens - comp_tokens
        compression_ratio = 1 - (comp_tokens / orig_tokens) if orig_tokens > 0 else 0
        
        # Calcular similitud semántica usando el modelo (alternativa a embeddings)
        try:
            # Usar el modelo para evaluar similitud semántica
            eval_response = self._ollama_chat(
                messages=[
                    {"role": "system", "content": "Eres un evaluador de similitud semántica. Responde solo con un número entre 0 y 1."},
                    {"role": "user", "content": f"¿Qué tan similar es el significado de estas dos frases?\nFrase 1: {original}\nFrase 2: {compressed}\nResponde solo con un número entre 0 y 1 (1 = idéntico, 0 = sin relación):"}
                ],
                model=self.config.evaluator_model,
                temperature=0.0,
                think=False,
                num_predict=16,
            )
            semantic_similarity = self._parse_semantic_score(
                eval_response["message"]["content"].strip()
            )
        except Exception as e:
            logger.warning(f"No se pudo calcular similitud semántica: {e}")
            semantic_similarity = 0.0
        
        # Evaluar calidad
        if semantic_similarity >= 0.95:
            evaluation = "excellent"
        elif semantic_similarity >= 0.85:
            evaluation = "good"
        elif semantic_similarity >= 0.70:
            evaluation = "fair"
        else:
            evaluation = "poor"
        
        return ComparisonResult(
            original=original,
            compressed=compressed,
            semantic_similarity=float(semantic_similarity),
            token_savings=token_savings,
            compression_ratio=compression_ratio,
            evaluation=evaluation
        )
    
    def batch_compress(
        self,
        prompts: list[str],
        strategy: Optional[str] = None
    ) -> list[CompressionResult]:
        """
        Comprime múltiples prompts.
        
        Args:
            prompts: Lista de prompts a comprimir
            strategy: Estrategia de compresión
            
        Returns:
            Lista de CompressionResult
        """
        results = []
        for prompt in prompts:
            try:
                result = self.compress(prompt, strategy=strategy)
                results.append(result)
            except Exception as e:
                logger.error(f"Error comprimiendo prompt: {e}")
                # Crear resultado de error
                results.append(CompressionResult(
                    original_prompt=prompt,
                    compressed_prompt="",
                    original_tokens=self._count_tokens(prompt),
                    compressed_tokens=0,
                    compression_ratio=0,
                    processing_time_ms=0,
                    metadata={"error": str(e)}
                ))
        return results
