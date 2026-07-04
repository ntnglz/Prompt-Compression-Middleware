# Benchmark PCM — example_prompts.json

- **Generado:** 2026-07-03T21:01:07.952086+00:00
- **Modelo:** `granite4.1:3b`
- **Estrategia:** `balanced`
- **Thinking:** `auto/off`
- **Similitud semántica (LLM):** sí

## Resumen

| Métrica | Valor |
|---------|-------|
| Prompts | 10 |
| Ratio medio | 40.08% |
| Ratio min / max | 12.90% / 58.97% |
| Tokens ahorrados | 145 |
| Similitud formato (media) | 100.00% |
| Similitud semántica (media) | 90.00% |
| Tiempo medio | 959 ms |

## Detalle por prompt

| ID | Categoría | Ratio | Formato | Semántica | Original → Comprimido |
|----|-----------|-------|---------|-----------|------------------------|
| prompt_001 | code_review | 50% | 100% | 90% | Analiza cuidadosamente este código Pytho... → `TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity` |
| prompt_002 | translation | 53% | 100% | 90% | Por favor, podrías traducir este documen... → `TASK=translate FROM=es TO=en STYLE=formal DOMAIN=technical` |
| prompt_003 | summarization | 59% | 100% | 90% | Necesito que generes un resumen ejecutiv... → `TASK=summarize INPUT=annual_report FORMAT=list HIGHLIGHT=key_points,metrics` |
| prompt_004 | code_review | 48% | 100% | 90% | Generate a comprehensive analysis of thi... → `TASK=analyze INPUT=javascript CHECK=security,perf,code_smells FORMAT=json` |
| prompt_005 | explanation | 34% | 100% | 90% | Explica el concepto de Inteligencia Arti... → `TASK=explain TOPIC=AI AUDIENCE=child_10 STYLE=simple USE=analogies,examples` |
| prompt_006 | writing | 28% | 100% | 90% | I need you to write a professional email... → `TASK=write TYPE=email TO=client TOPIC=AI_data_service STYLE=professional TONE=concise INCLUDE=benefits,cta` |
| prompt_007 | comparison | 38% | 100% | 90% | Comparar estos dos algoritmos de ordenam... → `TASK=compare ITEMS=quicksort,mergesort CRITERIA=time_complexity,space_complexity,use_cases` |
| prompt_008 | planning | 26% | 100% | 90% | Create a detailed project plan for devel... → `TASK=create_plan TYPE=project DOMAIN=mobile_app FEATURES=auth,realtime_chat,image_sharing` |
| prompt_009 | legal_review | 51% | 100% | 90% | Revisa este contrato legal y señala toda... → `TASK=review INPUT=contract FOCUS=problematic_clauses CHECK=cancellation_terms,penalties` |
| prompt_010 | database_design | 13% | 100% | 90% | Design a database schema for an e-commer... → `TASK=design SCHEMA=ecommerce ENTITIES=users,products,orders,reviews REQUIRE=relationships,indexes OPTIMIZE=performance` |

## Esperado vs obtenido

### prompt_001 (code_review)

**Esperado:** `TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity`

**Obtenido:** `TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity`


### prompt_002 (translation)

**Esperado:** `TASK=translate FROM=es TO=en STYLE=formal DOMAIN=technical`

**Obtenido:** `TASK=translate FROM=es TO=en STYLE=formal DOMAIN=technical`


### prompt_003 (summarization)

**Esperado:** `TASK=summarize INPUT=annual_report FORMAT=list HIGHLIGHT=key_points,metrics`

**Obtenido:** `TASK=summarize INPUT=annual_report FORMAT=list HIGHLIGHT=key_points,metrics`


### prompt_004 (code_review)

**Esperado:** `TASK=analyze INPUT=javascript CHECK=security,perf,code_smells FORMAT=json`

**Obtenido:** `TASK=analyze INPUT=javascript CHECK=security,perf,code_smells FORMAT=json`


### prompt_005 (explanation)

**Esperado:** `TASK=explain TOPIC=AI AUDIENCE=child_10 STYLE=simple USE=analogies,examples`

**Obtenido:** `TASK=explain TOPIC=AI AUDIENCE=child_10 STYLE=simple USE=analogies,examples`


### prompt_006 (writing)

**Esperado:** `TASK=write TYPE=email TO=client TOPIC=AI_data_service STYLE=professional TONE=concise INCLUDE=benefits,cta`

**Obtenido:** `TASK=write TYPE=email TO=client TOPIC=AI_data_service STYLE=professional TONE=concise INCLUDE=benefits,cta`


### prompt_007 (comparison)

**Esperado:** `TASK=compare ITEMS=quicksort,mergesort CRITERIA=time_complexity,space_complexity,use_cases`

**Obtenido:** `TASK=compare ITEMS=quicksort,mergesort CRITERIA=time_complexity,space_complexity,use_cases`


### prompt_008 (planning)

**Esperado:** `TASK=create_plan TYPE=project DOMAIN=mobile_app FEATURES=auth,realtime_chat,image_sharing`

**Obtenido:** `TASK=create_plan TYPE=project DOMAIN=mobile_app FEATURES=auth,realtime_chat,image_sharing`


### prompt_009 (legal_review)

**Esperado:** `TASK=review INPUT=contract FOCUS=problematic_clauses CHECK=cancellation_terms,penalties`

**Obtenido:** `TASK=review INPUT=contract FOCUS=problematic_clauses CHECK=cancellation_terms,penalties`


### prompt_010 (database_design)

**Esperado:** `TASK=design SCHEMA=ecommerce ENTITIES=users,products,orders,reviews REQUIRE=relationships,indexes OPTIMIZE=performance`

**Obtenido:** `TASK=design SCHEMA=ecommerce ENTITIES=users,products,orders,reviews REQUIRE=relationships,indexes OPTIMIZE=performance`
