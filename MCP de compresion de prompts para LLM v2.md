---

# Roadmap de implementación

El proyecto puede abordarse de forma incremental, validando cada etapa antes de aumentar la complejidad.

## Fase 1 – Prototipo

Objetivo: demostrar que la compresión de prompts aporta beneficios reales.

Implementación:

```
Usuario
    ↓
LLM pequeño (Ollama)
    ↓
LLM principal
```

El LLM pequeño recibe instrucciones para:

- Reducir el número de tokens.
- Mantener exactamente la intención del prompt.
- No resumir, sino reescribir.
- Utilizar estructuras compactas cuando sea posible.

Modelos candidatos:

- Qwen 2.5 / Qwen 3 (1.5B–4B)
- Gemma 3 (4B)
- Phi-4 Mini
- Llama 3.x (3B)

Métricas:

- Ratio de compresión.
- Tiempo adicional introducido.
- Coste total.
- Calidad de la respuesta final.

---

## Fase 2 – Compresor especializado

Una vez validado el concepto, sustituir el LLM genérico por un modelo afinado (LoRA o Fine-Tuning) cuya única tarea sea la compresión de prompts.

Entradas:

```
Prompt original
```

Salidas:

```
Prompt comprimido
```

El modelo aprendería patrones de reescritura mucho más eficientes que un prompt estático.

---

## Fase 3 – Entrenamiento mediante RL

Optimizar el compresor utilizando un modelo grande como evaluador.

Proceso:

```
Prompt original
        │
        ▼
Compresor
        │
        ▼
Prompt comprimido
        │
        ├──────────────► LLM
        │                    │
        │                    ▼
Prompt original ───────► LLM
                             │
                             ▼
                  Comparación automática
```

La función de recompensa puede combinar:

- Máxima reducción de tokens.
- Mínima pérdida semántica.
- Máxima similitud entre respuestas.

De esta forma, el compresor aprendería automáticamente nuevas estrategias de representación.

---

## Fase 4 – Lenguaje intermedio (LLM Intermediate Representation)

La hipótesis más ambiciosa del proyecto es que el mejor formato para comunicarse con un LLM no sea el lenguaje natural.

El compresor podría desarrollar una representación intermedia específica (LLM IR), análoga a las representaciones intermedias utilizadas por los compiladores tradicionales (por ejemplo, LLVM IR).

Características deseables:

- Muy compacta.
- Semánticamente estable.
- Independiente del idioma humano.
- Fácilmente interpretable por distintos LLM.
- Optimizable para modelos concretos.

Ejemplo conceptual:

```
task(review){
 input:swift
 bugs
 perf
 security
 out:md
}
```

Esta representación no estaría diseñada para ser leída por personas, sino para maximizar la eficiencia de los modelos.

---

# Visión a largo plazo

El objetivo final del proyecto no es únicamente reducir tokens.

La visión consiste en introducir una nueva capa de infraestructura para los ecosistemas de IA, equivalente al papel que desempeñan los compiladores en la ingeniería del software.

Así como un compilador transforma un lenguaje de alto nivel en una representación optimizada para una CPU, el Prompt Compression Middleware transformaría el lenguaje natural en una representación optimizada para un LLM determinado.

En este escenario, la optimización dejaría de centrarse exclusivamente en los modelos y pasaría también a optimizar el protocolo de comunicación entre agentes y modelos de lenguaje.