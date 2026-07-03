"""
Test del compresor de prompts PCM

Ejecuta con: python -m pytest tests/test_compressor.py -v
"""

import pytest
import time
from pcm.compressor import PromptCompressor, CompressorConfig
from pcm.models import CompressionResult, ComparisonResult


# Fixtures
@pytest.fixture
def compressor():
    """Crea un compresor con configuración de prueba"""
    return PromptCompressor(
        config=CompressorConfig(
            model="qwen3:4b",
            temperature=0.1,
            timeout=30
        )
    )


class TestCompressor:
    """Tests para el compresor de prompts"""
    
    def test_compressor_initialization(self):
        """Test que el compresor se inicializa correctamente"""
        compressor = PromptCompressor()
        assert compressor.config is not None
        assert compressor.config.model == "granite4.1:3b"
        assert compressor.config.temperature == 0.1
    
    def test_compressor_with_custom_config(self):
        """Test con configuración personalizada"""
        config = CompressorConfig(
            model="phi3:3.8b-mini-instruct",
            temperature=0.3,
            strategy="aggressive"
        )
        compressor = PromptCompressor(config=config)
        assert compressor.config.model == "phi3:3.8b-mini-instruct"
        assert compressor.config.temperature == 0.3
    
    def test_invalid_temperature(self):
        """Test que falla con temperatura inválida"""
        config = CompressorConfig(temperature=1.5)
        with pytest.raises(ValueError):
            PromptCompressor(config=config)
    
    def test_invalid_timeout(self):
        """Test que falla con timeout inválido"""
        config = CompressorConfig(timeout=-1)
        with pytest.raises(ValueError):
            PromptCompressor(config=config)

    def test_compression_prompt_includes_strategy(self):
        """Test que la estrategia modifica el system prompt"""
        compressor = PromptCompressor()
        balanced = compressor._get_compression_prompt("balanced")
        aggressive = compressor._get_compression_prompt("aggressive")
        conservative = compressor._get_compression_prompt("conservative")

        assert "ENFATIZA" not in balanced
        assert "GLOSARIO PCM" in balanced
        assert "Reducción máxima" in aggressive
        assert "Preserva cada detalle" in conservative

    def test_resolve_think_for_qwen3(self):
        """Qwen3 usa thinking por defecto; otros modelos no."""
        compressor = PromptCompressor()
        assert compressor._resolve_think("qwen3:4b") is True
        assert compressor._resolve_think("Qwen3:1.7b") is True
        assert compressor._resolve_think("gemma3:4b") is None
        assert compressor._resolve_think("granite4.1:3b") is None

    def test_force_no_think_overrides_auto(self):
        compressor = PromptCompressor(CompressorConfig(model="qwen3:4b", think=False))
        assert compressor._resolve_think() is False

    def test_extract_pcm_line(self):
        compressor = PromptCompressor()
        noisy = (
            "Okay, let's think...\n"
            "TASK=review INPUT=python CHECK=race,leak,perf\n"
            "extra"
        )
        assert compressor._extract_pcm_line(noisy) == (
            "TASK=review INPUT=python CHECK=race,leak,perf"
        )

    def test_parse_semantic_score(self):
        compressor = PromptCompressor()
        assert compressor._parse_semantic_score("0.95") == 0.95
        assert compressor._parse_semantic_score("blah blah\n0.85") == 0.85


class TestCompression:
    """Tests para la funcionalidad de compresión"""
    
    def test_compress_simple_prompt(self, compressor):
        """Test compresión de un prompt simple"""
        prompt = "Analiza este código Python"
        result = compressor.compress(prompt)
        
        assert isinstance(result, CompressionResult)
        assert result.original_prompt == prompt
        assert result.compressed_prompt != ""
        assert result.original_tokens > 0
        assert result.compressed_tokens >= 0
        assert 0 <= result.compression_ratio <= 1
        assert result.processing_time_ms > 0
    
    def test_compress_with_strategy(self, compressor):
        """Test compresión con diferentes estrategias"""
        prompt = "Por favor, analiza este código cuidadosamente"
        
        # Agresivo
        result_aggressive = compressor.compress(prompt, strategy="aggressive")
        assert result_aggressive.strategy == "aggressive"
        
        # Balanceado
        result_balanced = compressor.compress(prompt, strategy="balanced")
        assert result_balanced.strategy == "balanced"
        
        # Conservador
        result_conservative = compressor.compress(prompt, strategy="conservative")
        assert result_conservative.strategy == "conservative"
    
    def test_compress_preserves_intent(self, compressor):
        """Test que la compresión preserva la intención"""
        # Prompts con la misma intención deben producir resultados similares
        prompt1 = "Analiza este código Python buscando errores"
        prompt2 = "Revisa este código en Python para encontrar errores"
        
        result1 = compressor.compress(prompt1)
        result2 = compressor.compress(prompt2)
        
        # Ambos deben tener ratio de compresión positivo
        assert result1.compression_ratio > 0
        assert result2.compression_ratio > 0
    
    def test_compress_long_prompt(self, compressor):
        """Test compresión de un prompt largo"""
        prompt = """
        Por favor, podrías analizar este código fuente de Python de manera 
        muy detallada y exhaustiva, buscando cualquier tipo de problema 
        que pueda existir, incluyendo pero no limitado a: errores de sintaxis, 
        problemas de rendimiento, fugas de memoria, condiciones de carrera, 
        vulnerabilidades de seguridad, y cualquier otra literatura anti-patrón. 
        Además, por favor, genera un informe completo en formato Markdown que 
        incluya todas tus hallazgos organizados por categorías y severidad.
        """
        result = compressor.compress(prompt)
        
        assert isinstance(result, CompressionResult)
        # Un prompt largo debería tener un buen ratio de compresión
        assert result.compression_ratio > 0.3  # Al menos 30%
        assert result.compressed_tokens < result.original_tokens


class TestComparison:
    """Tests para la comparación de prompts"""
    
    def test_compare_prompts(self, compressor):
        """Test comparación de prompts"""
        original = "Analiza este código"
        compressed = "TASK=analyze INPUT=code"
        
        result = compressor.compare_prompts(original, compressed)
        
        assert isinstance(result, ComparisonResult)
        assert result.original == original
        assert result.compressed == compressed
        assert 0 <= result.semantic_similarity <= 1
        assert result.token_savings >= 0
        assert 0 <= result.compression_ratio <= 1
    
    def test_compare_with_auto_compression(self, compressor):
        """Test comparación con compresión automática"""
        original = "Traduce este texto al inglés"
        
        result = compressor.compare_prompts(original)
        
        assert isinstance(result, ComparisonResult)
        assert result.original == original
        assert result.compressed != ""


class TestBatch:
    """Tests para compresión en batch"""
    
    def test_batch_compress(self, compressor):
        """Test compresión de múltiples prompts"""
        prompts = [
            "Analiza este código",
            "Traduce al inglés",
            "Resume este texto",
            "Genera un informe"
        ]
        
        results = compressor.batch_compress(prompts)
        
        assert len(results) == len(prompts)
        for result in results:
            assert isinstance(result, CompressionResult)
            assert result.compressed_prompt != ""


class TestTokenCounting:
    """Tests para el conteo de tokens"""
    
    def test_token_counting(self, compressor):
        """Test que el conteo de tokens funciona"""
        text = "Hola mundo"
        count = compressor._count_tokens(text)
        
        assert isinstance(count, int)
        assert count > 0
    
    def test_token_counting_long_text(self, compressor):
        """Test conteo de tokens en texto largo"""
        text = " " * 1000  # 1000 espacios
        count = compressor._count_tokens(text)
        
        # Debería ser aproximadamente proporcional
        assert count > 0


class TestPerformance:
    """Tests de rendimiento"""
    
    def test_compression_speed(self, compressor):
        """Test que la compresión no es demasiado lenta"""
        prompt = "Analiza este código"
        
        start_time = time.time()
        result = compressor.compress(prompt)
        elapsed = time.time() - start_time
        
        # No debería tardar más de 10 segundos en un prompt simple
        assert elapsed < 10.0
        assert result.processing_time_ms < 10000


# Marca estos tests para que se ejecuten en orden
@pytest.mark.order1
class TestIntegration:
    """Tests de integración"""
    
    def test_full_workflow(self, compressor):
        """Test un flujo completo de compresión"""
        # 1. Comprimir
        original = "Genera un resumen detallado de este documento en Markdown"
        result = compressor.compress(original)
        
        # 2. Comparar
        comparison = compressor.compare_prompts(
            original, 
            result.compressed_prompt
        )
        
        # 3. Verificar
        assert result.original_tokens > result.compressed_tokens
        assert comparison.semantic_similarity > 0.5
        assert comparison.token_savings > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
